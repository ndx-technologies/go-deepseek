package godeepseek_test

import (
	"net/http"
	"os"
	"testing"

	godeepseek "github.com/ndx-technologies/go-deepseek"
)

func TestClient_ChatCompletions(t *testing.T) {
	if testing.Short() {
		t.Skip("network")
	}

	var config godeepseek.Config
	config = config.WithDefaults()

	var secrets godeepseek.Secrets
	secrets.Token = os.Getenv("DEEPSEEK_TOKEN")

	client := godeepseek.Client{Config: config, Secrets: secrets, HTTPClient: http.DefaultClient}

	t.Run("hello", func(t *testing.T) {
		req := godeepseek.ChatCompletionRequest{
			Messages: []godeepseek.Message{
				{Content: "Say hi", Role: godeepseek.User},
			},
			MaxTokens: 1000,
			Model:     godeepseek.DeepSeekV4Flash,
		}

		resp, err := client.ChatCompletions(t.Context(), req)
		if err != nil {
			t.Error(err)
		}

		t.Log(resp)

		if resp == nil {
			t.Error("got nil response")
		}
		if len(resp.Choices) == 0 {
			t.Error("got empty choices")
		}
		if got := resp.Choices[0].Message.Content; got == "" {
			t.Error("got empty content")
		}
	})

	t.Run("tool", func(t *testing.T) {
		terminalTool := godeepseek.Tool{
			Type: godeepseek.FunctionToolType,
			Function: godeepseek.Function{
				Name:        "terminal",
				Description: "Run a shell command in the terminal",
				Parameters: godeepseek.FunctionParameters{
					"type": "object",
					"properties": map[string]any{
						"command": map[string]any{
							"type":        "string",
							"description": "the command to run",
						},
					},
					"required": []string{"command"},
				},
			},
		}

		req := godeepseek.ChatCompletionRequest{
			Messages: []godeepseek.Message{
				{Role: godeepseek.System, Content: "You are testing assistant. Perform simple single test call for tools user asks."},
				{Role: godeepseek.User, Content: "lets test terminal"},
			},
			Model:     godeepseek.DeepSeekV4Flash,
			MaxTokens: 1000,
			Tools:     []godeepseek.Tool{terminalTool},
		}

		// hop 1: model should respond with a tool call
		resp, err := client.ChatCompletions(t.Context(), req)
		if err != nil {
			t.Fatal(err)
		}

		if resp == nil || len(resp.Choices) == 0 {
			t.Fatal("got empty response")
		}

		choice := resp.Choices[0]
		t.Logf("hop1: finish=%s content=%q usage=%+v", choice.FinishReason, choice.Message.Content, resp.Usage)

		if len(choice.Message.ToolCalls) == 0 {
			t.Fatalf("expected tool call, got finish=%s content=%q", choice.FinishReason, choice.Message.Content)
		}

		if resp.Usage.CompletionTokens == 0 {
			t.Error("usage tokens empty: copmletion")
		}
		if resp.Usage.PromptTokens == 0 {
			t.Error("usage tokens empty: prompt")
		}
		if resp.Usage.TotalTokens == 0 {
			t.Error("usage tokens empty: total")
		}
		if resp.Usage.TotalTokens != (resp.Usage.PromptTokens + resp.Usage.CompletionTokens) {
			t.Error(resp.Usage.TotalTokens, "!=", resp.Usage.PromptTokens, "+", resp.Usage.CompletionTokens)
		}

		var call *godeepseek.ToolCall
		for i := range choice.Message.ToolCalls {
			if choice.Message.ToolCalls[i].Function.Name == "terminal" {
				call = &choice.Message.ToolCalls[i]
				break
			}
		}
		if call == nil {
			t.Fatalf("expected terminal tool call, got %+v", choice.Message.ToolCalls)
		}
		t.Logf("tool call: id=%s name=%s args=%s", call.ID, call.Function.Name, call.Function.Arguments)

		// hop 2: send the tool result back and get the final answer
		messages := append(append([]godeepseek.Message{}, req.Messages...), choice.Message)
		messages = append(messages, godeepseek.Message{
			Role:       godeepseek.ToolRole,
			ToolCallID: call.ID,
			Content:    `{"result":"not available"}`,
		})

		resp2, err := client.ChatCompletions(t.Context(), godeepseek.ChatCompletionRequest{
			Messages:  messages,
			Model:     godeepseek.DeepSeekV4Flash,
			MaxTokens: 1000,
		})
		if err != nil {
			t.Fatal(err)
		}

		if resp2 == nil || len(resp2.Choices) == 0 {
			t.Fatal("got empty final response")
		}
		t.Logf("hop2: finish=%s content=%q usage=%+v", resp2.Choices[0].FinishReason, resp2.Choices[0].Message.Content, resp2.Usage)

		if got := resp2.Choices[0].Message.Content; got == "" {
			t.Error("got empty final content")
		}
	})
}

func TestClient_Models(t *testing.T) {
	if testing.Short() {
		t.Skip("network")
	}

	var config godeepseek.Config
	config = config.WithDefaults()

	var secrets godeepseek.Secrets
	secrets.Token = os.Getenv("DEEPSEEK_TOKEN")

	client := godeepseek.Client{Config: config, Secrets: secrets, HTTPClient: http.DefaultClient}

	models, err := client.Models(t.Context())
	if err != nil {
		t.Error(err)
	}

	t.Log(models)

	if len(models) == 0 {
		t.Error("got empty models")
	}
}

func TestClient_UserBalance(t *testing.T) {
	if testing.Short() {
		t.Skip("network")
	}

	var config godeepseek.Config
	config = config.WithDefaults()

	var secrets godeepseek.Secrets
	secrets.Token = os.Getenv("DEEPSEEK_TOKEN")

	client := godeepseek.Client{Config: config, Secrets: secrets, HTTPClient: http.DefaultClient}

	balance, err := client.UserBalance(t.Context())
	if err != nil {
		t.Error(err)
	}

	t.Log(balance)

	if balance == nil {
		t.Error("got nil balance")
	}
	if !balance.IsAvailable {
		t.Error("balance is not available")
	}
}
