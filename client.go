package godeepseek

import (
	"bytes"
	"context"
	"encoding/json/v2"
	"io"
	"log/slog"
	"net/http"
)

type Config struct {
	BaseURL string `json:"base_url"`
}

type Secrets struct {
	Token string `json:"token"`
}

func (s Config) WithDefaults() Config {
	if s.BaseURL == "" {
		s.BaseURL = "https://api.deepseek.com"
	}
	return s
}

// Client implements deepseek API HTTP client.
type Client struct {
	Config     Config
	Secrets    Secrets
	HTTPClient *http.Client
}

// ChatCompletion generates next chat message based on chat history.
// For higher cache hit rates do not change generation parameters or model once chat started.
// https://api-docs.deepseek.com/api/create-chat-completion
func (s Client) ChatCompletions(ctx context.Context, r ChatCompletionRequest) (*ChatCompletionResponse, error) {
	payload, err := json.Marshal(r)
	if err != nil {
		return nil, err
	}

	url := s.Config.BaseURL + "/chat/completions"
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(payload))
	if err != nil {
		return nil, err
	}

	req.Header.Add("Content-Type", "application/json")
	req.Header.Add("Accept", "application/json")
	req.Header.Add("Authorization", "Bearer "+s.Secrets.Token)

	resp, err := s.HTTPClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, err := io.ReadAll(resp.Body)
		if err != nil {
			slog.ErrorContext(ctx, "cannot read body", "error", err)
		}
		return nil, &Error{HTTPStatusCode: resp.StatusCode, Body: string(body)}
	}

	var res ChatCompletionResponse
	if err := json.UnmarshalRead(resp.Body, &res); err != nil {
		return nil, err
	}

	return &res, nil
}

type ChatCompletionRequest struct {
	Messages        []Message       `json:"messages"`                  // A list of messages comprising the conversation so far.
	Model           ModelID         `json:"model"`                     //
	Thinking        ThinkingConfig  `json:"thinking,omitzero"`         // default "enabled"
	ReasoningEffort ReasoningEffort `json:"reasoning_effort,omitzero"` // default "high"
	MaxTokens       int             `json:"max_tokens,omitzero"`       // The maximum number of tokens that can be generated in the chat completion. The total length of input tokens and generated tokens is limited by the model's context length. Default values and limits: https://api-docs.deepseek.com/quick_start/pricing
	ResponseFormat  ResponseFormat  `json:"response_format,omitzero"`  // default "text"
	Stop            []string        `json:"stop,omitzero"`             // Up to 16 sequences where the API will stop generating further tokens.
	Temperature     *float64        `json:"temperature,omitzero"`      // default 1. possible values <=2. What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic. We generally recommend altering this or top_p but not both.
	TopP            *float64        `json:"top_p,omitzero"`            // default 1. possible values <=1. An alternative to sampling with temperature, called nucleus sampling, where the model considers the results of the tokens with top_p probability mass. So 0.1 means only the tokens comprising the top 10% probability mass are considered. We generally recommend altering this or temperature but not both.
	Tools           []Tool          `json:"tools,omitzero"`            // max of 128 tools supported
	ToolChoice      []ToolChoice    `json:"tool_choice,omitzero"`      // Controls which (if any) tool is called by the model. "none" is the default when no tools are present. "auto" is the default if tools are present.
	LogProbs        bool            `json:"logprobs,omitzero"`         // Whether to return log probabilities of the output tokens or not. If true, returns the log probabilities of each output token returned in the content of message.
	TopLogProbs     int             `json:"top_logprobs,omitzero"`     // An integer between 0 and 20 specifying the number of most likely tokens to return at each token position, each with an associated log probability. logprobs must be set to true if this parameter is used.

	// UserID can be used to distinguish user identities on your side to help us with content safety review.
	// UserID can be used for KVCache isolation for privacy management.
	// UserID can be used for scheduling isolation of users on your business side.
	// For more details, please refer to [Rate Limit & Isolation](https://api-docs.deepseek.com/quick_start/rate_limit).
	// Do not include user privacy information in this field.
	UserID UserID `json:"user_id,omitzero"`
}

type Message struct {
	Content string `json:"content"`
	Role    Role   `json:"role"`
	Name    string `json:"name,omitzero"` // An optional name for the participant. Provides the model information to differentiate between participants of the same role.

	// tool
	ToolCallID ToolCallID `json:"tool_call_id,omitzero"` // (request) Tool call that this message is responding to.
	ToolCalls  []ToolCall `json:"tool_calls,omitzero"`   // (response) The tool calls generated by the model.

	// assistant
	ReasoningContent string `json:"reasoning_content,omitzero"` // (responses). For thinking mode only. The reasoning contents of the assistant message, before the final answer.
}

type Role string

const (
	System    Role = "system"
	User      Role = "user"
	Assistant Role = "assistant"
	ToolRole  Role = "tool"
)

type ToolCallID string

type ToolCall struct {
	ID       ToolCallID   `json:"id"`
	Type     ToolType     `json:"type"`
	Function FunctionCall `json:"function"`
}

type FunctionCall struct {
	Name FunctionName `json:"name"`

	// The arguments to call the function with, as generated by the model in JSON format.
	// Note that the model does not always generate valid JSON, and may hallucinate parameters not defined by your function schema.
	// Validate the arguments in your code before calling your function.
	Arguments string `json:"arguments"`
}

type ModelID string

const (
	DeepSeekV4Flash ModelID = "deepseek-v4-flash"
	DeepSeekV4Pro   ModelID = "deepseek-v4-pro"
)

type ThinkingConfig struct {
	Type ThinkingType `json:"type,omitzero"`
}

type ThinkingType string

const (
	ThinkingEnabled  ThinkingType = "enabled"
	ThinkingDisabled ThinkingType = "disabled"
)

type ReasoningEffort string

const (
	// For compatibility, medium and xhigh are mapped to high.
	ReasoningEffortLow  ReasoningEffort = "low"
	ReasoningEffortHigh ReasoningEffort = "high"
	ReasoningEffortMax  ReasoningEffort = "max"
)

// ResponseFormat that the model must output.
type ResponseFormat struct {
	Type ResponseFormatType `json:"type"`
}

type ResponseFormatType string

const (
	// ResponseFormatJSON guarantees the message the model generates is valid JSON.
	// Important: When using JSON Output, you must also instruct the model to produce JSON yourself via a system or user message.
	// Without this, the model may generate an unending stream of whitespace until the generation reaches the token limit, resulting in a long-running and seemingly "stuck" request.
	// Also note that the message content may be partially cut off if finish_reason="length", which indicates the generation exceeded max_tokens or the conversation exceeded the max context length.
	ResponseFormatJSON ResponseFormatType = "json_object"
	ResponseFormatText ResponseFormatType = "text"
)

type Tool struct {
	Type     ToolType `json:"type"`
	Function Function `json:"function"`
}

type ToolType string

const (
	FunctionToolType ToolType = "function"
)

type Function struct {
	Description string             `json:"description"`         // a description of what the function does, used by the model to choose when and how to call the function
	Name        FunctionName       `json:"name"`                // the name of the function to be called
	Parameters  FunctionParameters `json:"parameters,omitzero"` // omitting parameters defines a function with an empty parameter list
	Strict      bool               `json:"strict,omitzero"`     // (beta) if true, then output always complies with the function's JSON schema
}

// FunctionName must be a-z, A-Z, 0-9, or contain underscores and dashes, with a maximum length of 64.
type FunctionName string

// FunctionParameters described as a JSON Schema object.
// See the Tool Calls Guide for examples, and the JSON Schema reference for documentation about the format.
// https://api-docs.deepseek.com/guides/tool_calls
// https://json-schema.org/understanding-json-schema/
type FunctionParameters map[string]any

type ToolChoice string

const (
	NoTools       ToolChoice = "none"     // model will not call any tool and instead generates a message
	AutoTools     ToolChoice = "auto"     // model can pick between generating a message or calling one or more tools
	RequiredTools ToolChoice = "required" // model must call one or more tools
)

// UserID allowed character set is [a-zA-Z0-9\-_], with a maximum length of 512.
type UserID string

type ChatCompletionResponse struct {
	ID                ChatCompletionID   `json:"id"`
	Choices           []CompletionChoice `json:"choices"`
	CreatedAt         int                `json:"created"` // The Unix timestamp (in seconds) of when the chat completion was created.
	Model             ModelID            `json:"model"`
	SystemFingerprint string             `json:"system_fingerprint"` // This fingerprint represents the backend configuration that the model runs with.
	Usage             Usage              `json:"usage"`              // Usage statistics for the completion request.
}

type ChatCompletionID string

type CompletionChoice struct {
	Index        int          `json:"index"`
	FinishReason FinishReason `json:"finish_reason"` // the reason the model stopped generating tokens
	Message      Message      `json:"message"`
	LogProbs     *LogProbs    `json:"logprobs"` // Log probability information for the choice.
}

type FinishReason string

const (
	Stop                       FinishReason = "stop"                         // model hit a natural stop point or a provided stop sequence
	FinishReasonLength         FinishReason = "length"                       // maximum number of tokens specified in the request was reached
	FinishReasonContentFilter  FinishReason = "content_filter"               // content was omitted due to a flag from our content filters
	FinishReasonToolCalls      FinishReason = "tool_calls"                   // model called a tool
	InsufficientSystemResource FinishReason = "insufficient_system_resource" // request is interrupted due to insufficient resource of the inference system
)

type Usage struct {
	CompletionTokens       int                    `json:"completion_tokens"`        // Number of tokens in the generated completion.
	PromptTokens           int                    `json:"prompt_tokens"`            // Number of tokens in the prompt. It equals prompt_cache_hit_tokens + prompt_cache_miss_tokens.
	PromptCacheHitTokens   int                    `json:"prompt_cache_hit_tokens"`  // Number of tokens in the prompt that hits the context cache.
	PromptCacheMissTokens  int                    `json:"prompt_cache_miss_tokens"` // Number of tokens in the prompt that misses the context cache.
	TotalTokens            int                    `json:"total_tokens"`             // Total number of tokens used in the request (prompt + completion).
	CompletionTokenDetails CompletionTokenDetails `json:"completion_tokens_details"`
}

type CompletionTokenDetails struct {
	ReasoningTokens int `json:"reasoning_tokens"`
}

type LogProbs struct {
	Content          []ContentLogProbs `json:"content"`           // A list of message content tokens with log probability information.
	ReasoningContent []ContentLogProbs `json:"reasoning_content"` // A list of message content tokens with log probability information.
}

type ContentLogProbs struct {
	TokenLogProbs `json:",inline"`
	TopLogProbs   []TokenLogProbs `json:"top_logprobs"` // List of the most likely tokens and their log probability, at this token position. In rare cases, there may be fewer than the number of requested top_logprobs returned.
}

type TokenLogProbs struct {
	Token   string  `json:"token"`
	LogProb float64 `json:"logprob"` // The log probability of this token, if it is within the top 20 most likely tokens. Otherwise, the value -9999.0 is used to signify that the token is very unlikely.
	Bytes   []int   `json:"bytes"`   // A list of integers representing the UTF-8 bytes representation of the token. Useful in instances where characters are represented by multiple tokens and their byte representations must be combined to generate the correct text representation. Can be null if there is no bytes representation for the token.
}

// Models currently available.
// https://api-docs.deepseek.com/api/list-models
func (s Client) Models(ctx context.Context) ([]Model, error) {
	url := s.Config.BaseURL + "/models"
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, err
	}

	req.Header.Add("Content-Type", "application/json")
	req.Header.Add("Accept", "application/json")
	req.Header.Add("Authorization", "Bearer "+s.Secrets.Token)

	resp, err := s.HTTPClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, err := io.ReadAll(resp.Body)
		if err != nil {
			slog.ErrorContext(ctx, "cannot read body", "error", err)
		}
		return nil, &Error{HTTPStatusCode: resp.StatusCode, Body: string(body)}
	}

	var res struct {
		Data []Model `json:"data"`
	}
	if err := json.UnmarshalRead(resp.Body, &res); err != nil {
		return nil, err
	}

	return res.Data, nil
}

type Model struct {
	ID      ModelID `json:"id"`
	OwnedBy string  `json:"owned_by"`
}

// UserBalance for current user.
// https://api-docs.deepseek.com/api/get-user-balance
func (s Client) UserBalance(ctx context.Context) (*UserBalance, error) {
	url := s.Config.BaseURL + "/user/balance"
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, err
	}

	req.Header.Add("Content-Type", "application/json")
	req.Header.Add("Accept", "application/json")
	req.Header.Add("Authorization", "Bearer "+s.Secrets.Token)

	resp, err := s.HTTPClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, err := io.ReadAll(resp.Body)
		if err != nil {
			slog.ErrorContext(ctx, "cannot read body", "error", err)
		}
		return nil, &Error{HTTPStatusCode: resp.StatusCode, Body: string(body)}
	}

	var res UserBalance
	if err := json.UnmarshalRead(resp.Body, &res); err != nil {
		return nil, err
	}

	return &res, nil
}

type UserBalance struct {
	IsAvailable  bool          `json:"is_available"` // Whether the user's balance is sufficient for API calls.
	BalanceInfos []BalanceInfo `json:"balance_infos"`
}

type BalanceInfo struct {
	Currency        Currency `json:"currency"`
	TotalBalance    string   `json:"total_balance"`     // The total available balance, including the granted balance and the topped-up balance.
	GrantedBalance  string   `json:"granted_balance"`   // The total not expired granted balance.
	ToppedUpBalance string   `json:"topped_up_balance"` // The total topped-up balance.
}

type Currency string

const (
	USD Currency = "USD"
	CNY Currency = "CNY"
)
