from typing import Literal, Optional, List, Dict, Any, Union

import time

import shortuuid
from pydantic import BaseModel, Field


class ErrorResponse(BaseModel):
    object: str = "error"
    message: str
    code: int


class ModelPermission(BaseModel):
    id: str = Field(default_factory=lambda: f"modelperm-{shortuuid.random()}")
    object: str = "model_permission"
    created: int = Field(default_factory=lambda: int(time.time()))
    allow_create_engine: bool = False
    allow_sampling: bool = True
    allow_logprobs: bool = True
    allow_search_indices: bool = True
    allow_view: bool = True
    allow_fine_tuning: bool = False
    organization: str = "*"
    group: Optional[str] = None
    is_blocking: str = False


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "fastchat"
    root: Optional[str] = None
    parent: Optional[str] = None
    permission: List[ModelPermission] = []


class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard] = []


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0
    completion_tokens: Optional[int] = 0


class APIChatCompletionRequest(BaseModel):
    model: str
    messages: Union[str, List[Dict[str, str]]]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    max_tokens: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None
    stream: Optional[bool] = False
    user: Optional[str] = None
    repetition_penalty: Optional[float] = 1.0


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Optional[Literal["stop", "length"]] = None


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{shortuuid.random()}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: UsageInfo


class DeltaMessage(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None


class ChatCompletionResponseStreamChoice(BaseModel):
    index: int
    delta: DeltaMessage
    finish_reason: Optional[Literal["stop", "length"]] = None


class ChatCompletionStreamResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{shortuuid.random()}")
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionResponseStreamChoice]


class APITokenCheckRequestItem(BaseModel):
    model: str
    prompt: str
    max_tokens: int


class APITokenCheckRequest(BaseModel):
    prompts: List[APITokenCheckRequestItem]


class APITokenCheckResponseItem(BaseModel):
    fits: bool
    tokenCount: int
    contextLength: int


class APITokenCheckResponse(BaseModel):
    prompts: List[APITokenCheckResponseItem]


class CompletionRequest(BaseModel):
    model: str
    prompt: Union[str, List[Any]]
    suffix: Optional[str] = None
    temperature: Optional[float] = 0.7
    n: Optional[int] = 1
    max_tokens: Optional[int] = 16
    stop: Optional[Union[str, List[str]]] = None
    stream: Optional[bool] = False
    top_p: Optional[float] = 1.0
    logprobs: Optional[int] = None
    echo: Optional[bool] = False
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    user: Optional[str] = None


class CompletionResponseChoice(BaseModel):
    index: int
    text: str
    logprobs: Optional[int] = None
    finish_reason: Optional[Literal["stop", "length"]] = None


class CompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"cmpl-{shortuuid.random()}")
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[CompletionResponseChoice]
    usage: UsageInfo


class CompletionResponseStreamChoice(BaseModel):
    index: int
    text: str
    logprobs: Optional[float] = None
    finish_reason: Optional[Literal["stop", "length"]] = None


class CompletionStreamResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"cmpl-{shortuuid.random()}")
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[CompletionResponseStreamChoice]

class APITextGenerationParam(BaseModel):
    best_of: Optional[int] = None
    decoder_input_details: Optional[bool] = None
    details: Optional[bool] = None
    do_sample: Optional[bool] = None
    max_new_tokens: Optional[int] = 512
    repetition_penalty: Optional[float] = None
    return_full_text: Optional[bool] = None
    seed: Optional[int] = None
    stop: Optional[List[str]] = None
    temperature: Optional[float] = 1.
    top_k: Optional[int] = 10.
    top_p: Optional[float] = None
    truncate: Optional[int] = None
    typical_p: Optional[float] = .95
    watermark: Optional[bool] = None


class APITextGenerationRequest(BaseModel):
    inputs: str
    parameters: Optional[APITextGenerationParam] = None

# {
#   "inputs": "My name is Olivier and I",
#   "parameters": {
#     "best_of": 1,
#     "decoder_input_details": true,
#     "details": true,
#     "do_sample": true,
#     "max_new_tokens": 20,
#     "repetition_penalty": 1.03,
#     "return_full_text": false,
#     "seed": null,
#     "stop": [
#       "photographer"
#     ],
#     "temperature": 0.5,
#     "top_k": 10,
#     "top_p": 0.95,
#     "truncate": null,
#     "typical_p": 0.95,
#     "watermark": true
#   }
# }