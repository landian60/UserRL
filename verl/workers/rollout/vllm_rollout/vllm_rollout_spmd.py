# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
The vllm_rollout that can be applied in different backend
When working with FSDP:
- Use DTensor weight loader (recommended) or HF weight loader
- Utilize state_dict from the FSDP to synchronize the weights among tp ranks in vLLM
When working with Megatron:
- Use Megatron weight loader
- During training, only the current pp stage holds the parameters
- Before inference, broadcast the parameters of the current pp rank
  to all other pp ranks (all pp ranks holds all the parameters)
- Bind the parameters to the inference engine
- Do inference in tp. pp is treated as additional dp
- After inference, all the parameters that doesn't belong to this pp rank is freed.
"""

import asyncio
import logging
import os
from contextlib import contextmanager
from copy import deepcopy
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

import numpy as np
import torch
import torch.distributed
from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDict
from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedTokenizer
from vllm import LLM, SamplingParams
from vllm.distributed import parallel_state as vllm_ps
from vllm.lora.request import LoRARequest
from vllm.worker.worker_base import WorkerWrapperBase

from verl import DataProto
from verl.third_party.vllm import vllm_version
from verl.utils.debug import GPUMemoryLogger
from verl.utils.torch_functional import get_response_mask, pad_2d_list_to_length, pad_sequence_to_length
from verl.workers.rollout.base import BaseRollout
from verl.workers.rollout.schemas import AsyncRolloutRequest, AsyncRolloutRequestStateEnum, FinishReasonTypeEnum
from verl.tools.base_tool import initialize_tools_from_config
from verl.tools.schemas import (
    OpenAIFunctionCallSchema,
    OpenAIFunctionParsedSchema,
    OpenAIFunctionToolCall,
    OpenAIFunctionToolSchema,
)

from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.srt.openai_api.protocol import Tool

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# TODO
# 1. support pp in vllm
# 2. passing tokenizer is not necessary? no encoding/decoding is happending here
# 3. simplify init logics


# NOTE(sgm): add for verl. We can optimize it by making the dataloader yield List[int] without padding.
def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor) -> List[int]:
    # remove the left padding in the prompt token_id
    # pad_token_id = self.llm_engine.tokenizer.pad_token_id if self.llm_engine.tokenizer.pad_token_id
    # is not None else self.llm_engine.tokenizer.eos_token_id
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    token_ids = prompt_token_ids[non_pad_index:].tolist()
    return token_ids


def _repeat_interleave(value: Union[torch.Tensor, np.ndarray], repeats: int) -> Union[torch.Tensor, List[Any]]:
    if isinstance(value, torch.Tensor):
        return value.repeat_interleave(repeats, dim=0)
    else:
        return np.repeat(value, repeats, axis=0)


def get_tool_call_parser_type(tokenizer: PreTrainedTokenizer) -> str:
    items = FunctionCallParser.ToolCallParserEnum.items()
    for parser_type, parser_cls in items:
        parser = parser_cls()
        if parser.bot_token.strip() in tokenizer.get_vocab() and (parser.eot_token == "" or parser.eot_token.strip() in tokenizer.get_vocab()):
            return parser_type
    raise ValueError(f"No tool call parser found for tokenizer {tokenizer}")


class vLLMRollout(BaseRollout):
    def __init__(self, model_path: str, config: DictConfig, tokenizer, model_hf_config, **kwargs):
        """A vLLM rollout. It requires the module is supported by the vllm.

        Args:
            module: module here follows huggingface APIs
            config: DictConfig
            tokenizer: the task/model tokenizer
            model_hf_config: the huggingface config to initiallize the generating model in vllm
            **kwargs: train_tp, for Megatron Backend to initialize hybrid engine (zero redundancy) process group
        """
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        assert not (not config.enforce_eager and config.free_cache_engine), "disable CUDA graph (enforce_eager = False) if free cache engine"

        tensor_parallel_size = self.config.get("tensor_model_parallel_size", 1)
        assert tensor_parallel_size <= torch.distributed.get_world_size(), "tensor parallel size should be less than or equal to the world size"
        max_num_batched_tokens = self.config.get("max_num_batched_tokens", 8192)

        if kwargs.get("train_tp") is not None:
            # deployed with megatron
            import os

            os.environ["CUDA_TIMER_STREAM_KAFKA_ENABLE"] = "0"
            os.environ["MEGATRON_IMPORT_TIMERS"] = "0"
            if vllm_version in (
                "0.5.4",
                "0.6.3",
            ):
                train_tp = kwargs.get("train_tp")
                num_tp_per_train_tp = train_tp // tensor_parallel_size
                vllm_ps.initialize_parallel_state(tensor_model_parallel_size=tensor_parallel_size, num_tp_per_train_tp=num_tp_per_train_tp)
            else:
                vllm_ps.initialize_model_parallel(tensor_model_parallel_size=tensor_parallel_size)

        rope_scaling_config = getattr(model_hf_config, "rope_scaling", None)
        if not rope_scaling_config:
            max_position_embeddings = None
            if hasattr(model_hf_config, "max_position_embeddings"):
                max_position_embeddings = model_hf_config.max_position_embeddings
            elif hasattr(model_hf_config, "llm_config") and hasattr(model_hf_config.llm_config, "max_position_embeddings"):
                max_position_embeddings = model_hf_config.llm_config.max_position_embeddings
            elif hasattr(model_hf_config, "text_config") and hasattr(model_hf_config.text_config, "max_position_embeddings"):
                max_position_embeddings = model_hf_config.text_config.max_position_embeddings
            if max_position_embeddings is None:
                raise ValueError("max_position_embeddings not found in model_hf_config")

            assert max_position_embeddings >= config.prompt_length + config.response_length, "model context length should be greater than total sequence length"

        max_model_len = int(config.max_model_len or config.prompt_length + config.response_length)
        self.max_model_len = max_model_len  # Save for use in multi-turn rollout

        if max_num_batched_tokens < max_model_len and self.config.enable_chunked_prefill:
            raise ValueError(
                "Enable chunked prefill, max_num_batched_tokens is smaller than max_model_len, \
                             please increase max_num_batched_tokens or disable chunked prefill"
            )

        trust_remote_code = kwargs.get("trust_remote_code", False)
        load_format = "dummy" if config.load_format.startswith("dummy") else config.load_format

        lora_kwargs = kwargs.pop("lora_kwargs", {})
        self.lora_kwargs = lora_kwargs
        # copy it to avoid secretly modifying the engine config
        engine_kwargs = {} if "engine_kwargs" not in config or "vllm" not in config.engine_kwargs else OmegaConf.to_container(deepcopy(config.engine_kwargs.vllm))
        # For each vLLM engine parameter,
        # - `None` means not setting it, so we pop it, and leave it to vLLM default value
        #    (which can vary across different vLLM versions);
        # - Otherwise it's the desired value we want to explicitly set.
        engine_kwargs = {key: val for key, val in engine_kwargs.items() if val is not None}
        is_multimodal = config.get("limit_images", None) is not None
        if is_multimodal:  # support for multi-image data
            engine_kwargs["limit_mm_per_prompt"] = {"image": config.get("limit_images")}

        llm_kwargs = {
            "model": model_path,
            "enable_sleep_mode": True,
            "tensor_parallel_size": tensor_parallel_size,
            "distributed_executor_backend": "external_launcher",
            "dtype": config.dtype,
            "enforce_eager": config.enforce_eager,
            "gpu_memory_utilization": config.gpu_memory_utilization,
            "disable_custom_all_reduce": True,
            "skip_tokenizer_init": False,
        }
        # Only set disable_mm_preprocessor_cache for multimodal models
        if is_multimodal:
            llm_kwargs["disable_mm_preprocessor_cache"] = True

        self.inference_engine = LLM(
            **llm_kwargs,
            max_model_len=max_model_len,
            load_format=load_format,
            disable_log_stats=config.disable_log_stats,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=config.enable_chunked_prefill,
            enable_prefix_caching=True,
            trust_remote_code=trust_remote_code,
            seed=config.get("seed", 0),
            **lora_kwargs,
            **engine_kwargs,
        )

        # Offload vllm model to reduce peak memory usage
        self.inference_engine.sleep(level=1)

        kwargs = dict(
            n=1,
            logprobs=0,  # can be set to 0 and let actor to recompute
            max_tokens=config.response_length,
        )

        # # we may detokenize the result all together later
        if vllm_version != "0.3.1":
            kwargs["detokenize"] = False

        # supporting adding any sampling params from the config file
        for k in config.keys():
            if hasattr(SamplingParams(), str(k)):
                kwargs[k] = config.get(k)

        print(f"kwargs: {kwargs}")
        self.sampling_params = SamplingParams(**kwargs)

        # Initialize pad_token_id and multi-turn tools
        self.pad_token_id = tokenizer.pad_token_id

        self._tool_schemas: List[OpenAIFunctionToolSchema] = []
        self._tool_map: Dict[str, Any] = {}
        self._tool_call_parser_type: Optional[str] = None
        self._function_call_parser: Optional[FunctionCallParser] = None
        if self.config.multi_turn.get('enable', False):
            (
                self._tool_schemas,
                self._tool_map,
                self._tool_call_parser_type,
                self._function_call_parser,
            ) = self._initialize_tools(config, tokenizer)

    def _build_lora_requests(self, batch_size: int):
        if not self.lora_kwargs:
            return None
        lora_int_ids = list(self.inference_engine.llm_engine.list_loras())
        if not lora_int_ids:
            return None
        lora_int_id = lora_int_ids[0]
        return [LoRARequest(lora_name=f'{lora_int_id}', lora_int_id=lora_int_id, lora_path='/simon-stub-path')] * batch_size

    def _initialize_tools(self, config: DictConfig, tokenizer: PreTrainedTokenizer):
        tool_config_path = config.multi_turn.get('tool_config_path', None)
        if tool_config_path is None:
            return [], {}, None, None

        tool_list = initialize_tools_from_config(tool_config_path)
        tool_schemas = [tool.get_openai_tool_schema() for tool in tool_list]
        tool_map = {tool.name: tool for tool in tool_list}
        tool_call_parser_type = get_tool_call_parser_type(tokenizer)
        sgl_tools = [Tool.model_validate(schema.model_dump()) for schema in tool_schemas]
        function_call_parser = FunctionCallParser(sgl_tools, tool_call_parser_type)
        return tool_schemas, tool_map, tool_call_parser_type, function_call_parser


    @contextmanager
    def update_sampling_params(self, **kwargs):
        # update sampling params
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)
        try:
            yield
        finally:
            # roll back to previous sampling params
            for key, value in old_sampling_params_args.items():
                setattr(self.sampling_params, key, value)

    @GPUMemoryLogger(role="vllm rollout spmd", logger=logger)
    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        if self.config.multi_turn.get('enable', False):
            return self._req_level_generate_sequences(prompts, **kwargs)

        # rebuild vllm cache engine
        if (
            vllm_version
            in (
                "0.5.4",
                "0.6.3",
            )
            and self.config.free_cache_engine
        ):
            self.inference_engine.init_cache_engine()

        idx = prompts.batch["input_ids"]  # (bs, prompt_length)
        # left-padded attention_mask
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]

        # used to construct attention_mask
        eos_token_id = prompts.meta_info["eos_token_id"]

        batch_size = idx.size(0)

        non_tensor_batch = prompts.non_tensor_batch
        if "raw_prompt_ids" not in non_tensor_batch:
            non_tensor_batch["raw_prompt_ids"] = np.array([_pre_process_inputs(self.pad_token_id, idx[i]) for i in range(batch_size)], dtype=object)

        if batch_size != len(non_tensor_batch["raw_prompt_ids"]):
            raise RuntimeError("vllm sharding manager is not work properly.")

        if "multi_modal_data" in non_tensor_batch:
            vllm_inputs = []
            for raw_prompt_ids, multi_modal_data in zip(non_tensor_batch.pop("raw_prompt_ids"), non_tensor_batch.pop("multi_modal_data")):
                vllm_inputs.append({"prompt_token_ids": raw_prompt_ids, "multi_modal_data": multi_modal_data})
        else:
            vllm_inputs = [{"prompt_token_ids": raw_prompt_ids} for raw_prompt_ids in non_tensor_batch.pop("raw_prompt_ids")]

        # ensure the type of `prompt_token_ids` passed to vllm is list[int]
        # https://github.com/volcengine/verl/pull/772
        for input_data in vllm_inputs:
            if isinstance(input_data["prompt_token_ids"], np.ndarray):
                input_data["prompt_token_ids"] = input_data["prompt_token_ids"].tolist()
            elif not isinstance(input_data["prompt_token_ids"], list):
                raise TypeError(f"prompt_token_ids must be a list or numpy array, got {type(input_data['prompt_token_ids'])}")

        do_sample = prompts.meta_info.get("do_sample", True)
        is_validate = prompts.meta_info.get("validate", False)
        if not do_sample:
            kwargs = {
                "best_of": 1,
                "top_p": 1.0,
                "top_k": -1,
                "min_p": 0.0,
                "temperature": 0,
                "n": 1,  # if greedy, only 1 response
            }
        elif is_validate:
            # TODO: try **
            kwargs = {
                "top_k": self.config.val_kwargs.top_k,
                "top_p": self.config.val_kwargs.top_p,
                "temperature": self.config.val_kwargs.temperature,
                "n": 1,  # if validate, already repeat in ray_trainer
            }

        lora_requests = None
        if self.lora_kwargs:
            lora_int_ids = list(self.inference_engine.llm_engine.list_loras())
            if len(lora_int_ids) > 0:
                lora_int_id = lora_int_ids[0]
                lora_requests = [LoRARequest(lora_name=f"{lora_int_id}", lora_int_id=lora_int_id, lora_path="/simon-stub-path")] * batch_size

        # users can customize different sampling_params at different run
        with self.update_sampling_params(**kwargs):
            outputs = self.inference_engine.generate(
                prompts=vllm_inputs,  # because we have already convert it to prompt token id
                sampling_params=self.sampling_params,
                lora_request=lora_requests,
                use_tqdm=False,
            )

            # TODO(sgm): disable logprob when recompute_log_prob is enable
            # if n = 1: (bs, response_length) ; if n > 1: (bs * n, response_length)

            response = []
            rollout_log_probs = []
            for output in outputs:
                for sample_id in range(len(output.outputs)):
                    response_ids = output.outputs[sample_id].token_ids
                    response.append(response_ids)
                    curr_log_prob = []
                    for i, logprob in enumerate(output.outputs[sample_id].logprobs):
                        curr_log_prob.append(logprob[response_ids[i]].logprob)
                    rollout_log_probs.append(curr_log_prob)

            response = pad_2d_list_to_length(response, self.pad_token_id, max_length=self.config.response_length).to(idx.device)
            rollout_log_probs = pad_2d_list_to_length(rollout_log_probs, -1, max_length=self.config.response_length).to(idx.device)
            rollout_log_probs = rollout_log_probs.to(torch.float32)

            if self.sampling_params.n > 1 and do_sample:
                idx = _repeat_interleave(idx, self.sampling_params.n)
                attention_mask = _repeat_interleave(attention_mask, self.sampling_params.n)
                position_ids = _repeat_interleave(position_ids, self.sampling_params.n)
                batch_size = batch_size * self.sampling_params.n
                # Expand all non_tensor_batch fields to match the expanded batch size
                # NOTE(linjunrong): for multi-turn https://github.com/volcengine/verl/pull/1037
                _non_tensor_batch = {}
                for key, val in non_tensor_batch.items():
                    _non_tensor_batch[key] = _repeat_interleave(val, self.sampling_params.n)
                non_tensor_batch = _non_tensor_batch

            seq = torch.cat([idx, response], dim=-1)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, 3, -1)

        # TODO(sgm): fix position_ids on right_pad
        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_attention_mask = get_response_mask(response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype)
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict(
            {
                "prompts": idx,
                "responses": response,
                "input_ids": seq,  # here input_ids become the whole sentences
                "rollout_log_probs": rollout_log_probs,  # we will recompute old log prob with actor
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )

        # free vllm cache engine
        if (
            vllm_version
            in (
                "0.5.4",
                "0.6.3",
            )
            and self.config.free_cache_engine
        ):
            self.inference_engine.free_cache_engine()

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)

    async def _semaphore_wrapped_rollout(self, sem, req, do_sample, is_validate, **kwargs):
        async with sem:
            return await self._async_rollout_a_request(req, do_sample, is_validate, **kwargs)

    @GPUMemoryLogger(role="vllm rollout spmd", logger=logger)
    @torch.no_grad()
    def _req_level_generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        do_sample = prompts.meta_info.get("do_sample", True)
        is_validate = prompts.meta_info.get("validate", False)
        tgt_device = prompts.batch["input_ids"].device

        req_list = self._preprocess_prompt_to_async_rollout_requests(
            prompts, n=1 if is_validate else self.config.n
        )
        if not req_list:
            return DataProto(batch=prompts.batch.clone(), non_tensor_batch=prompts.non_tensor_batch)

        max_async = self.config.multi_turn.get("max_async_requests", 64)
        sem = asyncio.Semaphore(max_async)
        loop = asyncio.get_event_loop()
        output_req_list = loop.run_until_complete(
            asyncio.gather(
                *[
                    self._semaphore_wrapped_rollout(sem, req, do_sample, is_validate, **kwargs)
                    for req in req_list
                ]
            )
        )

        sorted_output_req_list = sorted(output_req_list, key=lambda x: (x.batch_data_id, x.rollout_offset))

        prompt_ids, response_ids = [], []
        prompt_attention_mask, response_attention_mask = [], []
        prompt_position_ids, response_position_ids = [], []
        prompt_loss_mask, response_loss_mask = [], []
        turn_boundaries_list = []
        messages = []
        conversation_histories = []
        reward_scores = []

        for req in sorted_output_req_list:
            prompt_ids.append(torch.tensor(req.prompt_ids, dtype=torch.int, device=tgt_device))
            response_ids.append(torch.tensor(req.response_ids, dtype=torch.int, device=tgt_device))
            prompt_attention_mask.append(torch.tensor(req.prompt_attention_mask, dtype=torch.int, device=tgt_device))
            response_attention_mask.append(torch.tensor(req.response_attention_mask, dtype=torch.int, device=tgt_device))
            prompt_position_ids.append(torch.tensor(req.prompt_position_ids, dtype=torch.int, device=tgt_device))
            response_position_ids.append(torch.tensor(req.response_position_ids, dtype=torch.int, device=tgt_device))
            prompt_loss_mask.append(torch.tensor(req.prompt_loss_mask, dtype=torch.int, device=tgt_device))
            response_loss_mask.append(torch.tensor(req.response_loss_mask, dtype=torch.int, device=tgt_device))
            messages.append({"messages": req.messages})
            conversation_histories.append(req.conversation_histories)
            reward_scores.append(req.reward_scores)

            response_length = len(req.response_ids)
            turn_boundary_tensor = torch.zeros(response_length, dtype=torch.int, device=tgt_device)
            if req.turn_boundaries:
                prompt_length = len(req.prompt_ids)
                for boundary_pos in req.turn_boundaries:
                    response_pos = boundary_pos - prompt_length
                    if 0 <= response_pos < response_length:
                        turn_boundary_tensor[response_pos] = 1
                if turn_boundary_tensor.numel() > 0 and turn_boundary_tensor[0] == 0:
                    turn_boundary_tensor[0] = 1
            else:
                if response_length > 0:
                    turn_boundary_tensor[0] = 1
            turn_boundaries_list.append(turn_boundary_tensor)

        pad_token_id = self.pad_token_id if self.pad_token_id is not None else self.tokenizer.eos_token_id
        prompt_ids = pad_sequence(prompt_ids, batch_first=True, padding_value=pad_token_id)
        response_ids = pad_sequence(response_ids, batch_first=True, padding_value=pad_token_id)
        prompt_attention_mask = pad_sequence(prompt_attention_mask, batch_first=True, padding_value=0)
        response_attention_mask = pad_sequence(response_attention_mask, batch_first=True, padding_value=0)
        prompt_position_ids = pad_sequence(prompt_position_ids, batch_first=True, padding_value=0)
        response_position_ids = pad_sequence(response_position_ids, batch_first=True, padding_value=0)
        prompt_loss_mask = pad_sequence(prompt_loss_mask, batch_first=True, padding_value=0)
        response_loss_mask = pad_sequence(response_loss_mask, batch_first=True, padding_value=0)
        turn_boundaries = pad_sequence(turn_boundaries_list, batch_first=True, padding_value=0)

        if prompt_ids.shape[1] < self.config.prompt_length:
            prompt_ids = pad_sequence_to_length(prompt_ids, self.config.prompt_length, pad_token_id, left_pad=True)
            prompt_attention_mask = pad_sequence_to_length(prompt_attention_mask, self.config.prompt_length, 0, left_pad=True)
            prompt_position_ids = pad_sequence_to_length(prompt_position_ids, self.config.prompt_length, 0, left_pad=True)
            prompt_loss_mask = pad_sequence_to_length(prompt_loss_mask, self.config.prompt_length, 0, left_pad=True)
        if response_ids.shape[1] < self.config.response_length:
            response_ids = pad_sequence_to_length(response_ids, self.config.response_length, pad_token_id)
            response_attention_mask = pad_sequence_to_length(response_attention_mask, self.config.response_length, 0)
            response_position_ids = pad_sequence_to_length(response_position_ids, self.config.response_length, 0)
            response_loss_mask = pad_sequence_to_length(response_loss_mask, self.config.response_length, 0)
            turn_boundaries = pad_sequence_to_length(turn_boundaries, self.config.response_length, 0)

        input_ids = torch.cat((prompt_ids, response_ids), dim=-1)
        attention_mask = torch.cat((prompt_attention_mask, response_attention_mask), dim=-1)
        position_ids = torch.cat((prompt_position_ids, response_position_ids), dim=-1)
        loss_mask = torch.cat((prompt_loss_mask, response_loss_mask), dim=-1)

        batch = TensorDict(
            {
                "prompts": prompt_ids,
                "responses": response_ids,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "loss_mask": loss_mask,
                "turn_boundaries": turn_boundaries,
            },
            batch_size=len(sorted_output_req_list),
        )

        non_tensor_batch = {
            "messages": np.array(messages, dtype=object),
            "conversation_histories": np.array([[x] for x in conversation_histories], dtype=object),
            "reward_scores": np.array(reward_scores, dtype=object),
        }
        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)

    async def _async_rollout_a_request(self, req: AsyncRolloutRequest, do_sample: bool, is_validate: bool, **kwargs) -> AsyncRolloutRequest:
        _req = deepcopy(req)
        finish_reason_type = None
        turn_boundaries = []
        conversation_histories = []
        current_turns = 0

        while current_turns < self.config.multi_turn.max_turns:
            if _req.state == AsyncRolloutRequestStateEnum.PENDING:
                await self._handle_pending_state(_req)
                _req.state = AsyncRolloutRequestStateEnum.RUNNING
            elif _req.state == AsyncRolloutRequestStateEnum.TOOL_CALLING:
                if _req.messages[-1].tool_calls is not None:
                    parsed_tool_calls = _req.messages[-1].tool_calls
                    tool_call_results = await asyncio.gather(
                        *[
                            self._tool_map[tool_call.function.name].execute(
                                _req.request_id,
                                tool_call.function.arguments,
                                current_turns,
                                **_req.tools_kwargs[tool_call.function.name].get("execute_kwargs", {}),
                            )
                            for tool_call in parsed_tool_calls
                        ]
                    )
                    tool_responses = []
                    stop_conversation = False
                    for convo, tool_call in zip(tool_call_results, parsed_tool_calls):
                        if isinstance(convo, tuple):
                            resp = convo[0]
                            reward = convo[1] if len(convo) > 1 else 0.0
                            is_done = convo[2] if len(convo) > 2 else False
                            choice = convo[3] if len(convo) > 3 else ""
                            content = convo[4] if len(convo) > 4 else ""
                            metrics = convo[-1] if len(convo) > 5 else {}
                        else:
                            resp, reward, is_done, choice, content, metrics = convo, 0.0, False, "", "", {}
                        tool_responses.append(resp)
                        conversation_histories[-1]["choice"] = choice
                        conversation_histories[-1]["reward"] = reward
                        conversation_histories[-1]["content"] = content
                        _req.update_metrics(metrics, tool_call.function.name)
                        if is_done:
                            stop_conversation = True
                    _req.add_tool_response_messages(self.tokenizer, tool_responses)
                    if stop_conversation or len(_req.input_ids) >= self.max_model_len:
                        finish_reason_type = FinishReasonTypeEnum.STOP
                        break
                    _req.state = AsyncRolloutRequestStateEnum.RUNNING
                else:
                    raise ValueError("Unexpected tool calling state without tool calls")
            elif _req.state == AsyncRolloutRequestStateEnum.RUNNING:
                if len(_req.get_generation_prompt_ids(self.tokenizer)) + 1 >= self.max_model_len:
                    finish_reason_type = FinishReasonTypeEnum.LENGTH
                    break
                turn_boundaries.append(len(_req.input_ids))
                conversation_histories.append({"reward": 0.0, "choice": "action", "content": "", "turn_idx": len(conversation_histories)})
                output = await self._handle_engine_call(_req, do_sample, is_validate, **kwargs)
                content = output["text"]
                finish_reason_type = FinishReasonTypeEnum.from_str(output["meta_info"]["finish_reason"]["type"])
                current_turns += 1
                if finish_reason_type == FinishReasonTypeEnum.LENGTH:
                    _req.add_assistant_message(self.tokenizer, content)
                    break
                if self._function_call_parser and self._function_call_parser.has_tool_call(content):
                    finish_reason_type = FinishReasonTypeEnum.TOOL_CALL
                    _req.state = AsyncRolloutRequestStateEnum.TOOL_CALLING
                    try:
                        normed_content, tool_calls = self._function_call_parser.parse_non_stream(content)
                    except Exception:
                        normed_content, tool_calls = content, []
                    parsed_tool_calls = []
                    for tool_call in tool_calls:
                        function, has_decode_error = OpenAIFunctionCallSchema.from_openai_function_parsed_schema(
                            OpenAIFunctionParsedSchema(name=tool_call.name, arguments=tool_call.parameters)
                        )
                        if has_decode_error:
                            continue
                        parsed_tool_calls.append(
                            OpenAIFunctionToolCall(
                                id=str(tool_call.tool_index),
                                function=function,
                            )
                        )
                    if parsed_tool_calls:
                        _req.add_assistant_message(self.tokenizer, normed_content, tool_calls=parsed_tool_calls)
                    else:
                        _req.add_assistant_message(self.tokenizer, content)
                        finish_reason_type = FinishReasonTypeEnum.STOP
                        _req.state = AsyncRolloutRequestStateEnum.COMPLETED
                        break
                else:
                    _req.add_assistant_message(self.tokenizer, content)
                    break
            else:
                break

        if current_turns >= self.config.multi_turn.max_turns:
            finish_reason_type = FinishReasonTypeEnum.STOP

        async def calc_reward_and_release_fn(name: str, tool):
            reward = await tool.calc_reward(_req.request_id, **_req.tools_kwargs[name].get("calc_reward_kwargs", {}))
            await tool.release(_req.request_id, **_req.tools_kwargs[name].get("release_kwargs", {}))
            return name, reward

        if _req.tools_kwargs:
            tool_reward_scores = await asyncio.gather(
                *[calc_reward_and_release_fn(name, self._tool_map[name]) for name in _req.tools_kwargs.keys()]
            )
            tool_reward_scores = dict(tool_reward_scores)
        else:
            tool_reward_scores = {}

        _req.finalize(self.tokenizer, tool_reward_scores, turn_boundaries, conversation_histories, finish_reason_type)
        return _req

    async def _handle_engine_call(self, _req: AsyncRolloutRequest, do_sample: bool, is_validate: bool, override_n: bool = True, **kwargs):
        generation_prompt_ids = _req.get_generation_prompt_ids(self.tokenizer)
        max_new_tokens = min(self.config.response_length, self.max_model_len - len(generation_prompt_ids) - 1)
        if not do_sample:
            local_kwargs = {"n": 1, "temperature": 0, "top_p": 1, "top_k": -1}
        elif is_validate:
            local_kwargs = {"n": 1, "top_k": self.config.val_kwargs.top_k, "top_p": self.config.val_kwargs.top_p, "temperature": self.config.val_kwargs.temperature}
        else:
            local_kwargs = {"n": 1}
        local_kwargs["max_tokens"] = max_new_tokens
        if "n" in kwargs and not override_n:
            local_kwargs["n"] = kwargs["n"]

        def _run_generate():
            prompts = [{"prompt_token_ids": generation_prompt_ids}]
            lora_requests = self._build_lora_requests(len(prompts))
            with self.update_sampling_params(**local_kwargs):
                return self.inference_engine.generate(
                    prompts=prompts, sampling_params=self.sampling_params, lora_request=lora_requests, use_tqdm=False
                )

        loop = asyncio.get_running_loop()
        outputs = await loop.run_in_executor(None, _run_generate)
        request_output = outputs[0].outputs[0]
        text = getattr(request_output, "text", self.tokenizer.decode(request_output.token_ids, skip_special_tokens=True))
        finish_reason = getattr(request_output, "finish_reason", None) or "stop"
        return {"text": text, "meta_info": {"finish_reason": {"type": finish_reason}}}

    async def _handle_pending_state(self, _req: AsyncRolloutRequest) -> AsyncRolloutRequest:
        if _req.tool_schemas is None:
            return _req
        creation_tasks = []
        for tool_schema in _req.tool_schemas:
            tool = self._tool_map[tool_schema.function.name]
            create_kwargs = _req.tools_kwargs[tool.name].get("create_kwargs", {})
            create_kwargs.setdefault("max_turns", self.config.multi_turn.max_turns)
            create_kwargs.setdefault("model_name", self.config.multi_turn.get("model_name"))
            creation_tasks.append(tool.create(_req.request_id, **create_kwargs))
        if creation_tasks:
            await asyncio.gather(*creation_tasks)
        return _req

    def _preprocess_prompt_to_async_rollout_requests(self, prompts: DataProto, n: int) -> list[AsyncRolloutRequest]:
        if "raw_prompt" not in prompts.non_tensor_batch:
            raise ValueError("need data.return_raw_chat=True, due to no official way do parse_messages")
        req_list = []
        raw_prompts = prompts.non_tensor_batch["raw_prompt"]
        tools_kwargs_batch = prompts.non_tensor_batch.get("tools_kwargs", None)

        for data_idx, raw_prompt in enumerate(raw_prompts):
            for rollout_offset in range(n):
                if tools_kwargs_batch is not None:
                    _tools_kwargs = tools_kwargs_batch[data_idx]
                    _tool_schemas = [self._tool_map[name].get_openai_tool_schema() for name in _tools_kwargs.keys()]
                else:
                    _tools_kwargs = {}
                    _tool_schemas = None

                input_ids = _pre_process_inputs(self.pad_token_id, prompts.batch["input_ids"][data_idx])
                attention_mask = _pre_process_inputs(0, prompts.batch["attention_mask"][data_idx])

                req = AsyncRolloutRequest(
                    batch_data_id=data_idx,
                    rollout_offset=rollout_offset,
                    request_id=str(uuid4()),
                    state=AsyncRolloutRequestStateEnum.PENDING,
                    messages=raw_prompt.tolist(),
                    tool_schemas=_tool_schemas,
                    tools_kwargs=_tools_kwargs,
                    input_ids=input_ids,
                    response_ids=[],
                    attention_mask=attention_mask,
                    response_attention_mask=[],
                    response_position_ids=[],
                    response_loss_mask=[],
                    reward_scores={},
                    max_prompt_len=self.config.prompt_length,
                    max_response_len=self.config.response_length,
                    max_model_len=self.config.prompt_length + self.config.response_length if self.max_model_len is None else min(self.max_model_len, self.config.prompt_length + self.config.response_length),
                    use_inference_chat_template=self.config.multi_turn.use_inference_chat_template,
                    enable_tokenization_sanity_check=self.config.multi_turn.enable_tokenization_sanity_check,
                    tokenizer=self.tokenizer,
                )

                req_list.append(req)
        return req_list


class vLLMAsyncRollout:
    """vLLMAsyncRollout is a thin wrapper of WorkerWrapperBase,
    which is engine in single worker process.
    """

    def __init__(self, *args, **kwargs):
        # Engine is deferred to be initialized in init_worker
        self.inference_engine: WorkerWrapperBase = None
        self.sharding_manager = None
        self.is_sleep = False

    def init_worker(self, all_kwargs: List[Dict[str, Any]]):
        """Initialize worker engine."""
        all_kwargs[0]["rank"] = int(os.environ["RANK"])
        all_kwargs[0]["local_rank"] = 0

        self.vllm_config = all_kwargs[0]["vllm_config"]
        self.inference_engine = WorkerWrapperBase(vllm_config=self.vllm_config)
        self.inference_engine.init_worker(all_kwargs)

    def load_model(self, *args, **kwargs):
        self.inference_engine.load_model(*args, **kwargs)

        # inference engine is initialized now, update sharding manager
        self.sharding_manager.inference_engine = self.inference_engine
        self.sharding_manager.model_runner = self.inference_engine.worker.model_runner

    def sleep(self, *args, **kwargs):
        """Offload model weights and discard kv cache."""
        if self.is_sleep:
            return
        self.sharding_manager.__exit__(None, None, None)
        self.is_sleep = True

    def wake_up(self, *args, **kwargs):
        """Load model weights and build kv cache."""
        if not self.is_sleep:
            return
        self.sharding_manager.__enter__()  # pylint: disable=C2801
        self.is_sleep = False

    def execute_method(self, method: Union[str, bytes], *args, **kwargs):
        if method == "init_worker":
            return self.init_worker(*args, **kwargs)
        elif method == "load_model":
            return self.load_model(*args, **kwargs)
        elif method == "sleep":
            return self.sleep(*args, **kwargs)
        elif method == "wake_up":
            return self.wake_up(*args, **kwargs)
        else:
            return self.inference_engine.execute_method(method, *args, **kwargs)
