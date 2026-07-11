//! Prompt templates that turn chat messages into a model-specific prompt.
//!
//! Instruct-tuned GGUF models only behave when their training-time chat
//! markup is reproduced exactly. The template also decides which special
//! tokens appear in the prompt, so chat prompts are encoded without the
//! tokenizer's automatic special tokens.

use rust_ai_serving_engine_core::{EngineError, Result};

/// One conversation turn as accepted by the chat surfaces.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

/// Chat markup dialects supported by the bundled templates.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChatTemplate {
    /// `<|im_start|>` markup used by Qwen and many ChatML fine-tunes.
    ChatMl,
    /// Llama 3 header markup with `<|eot_id|>` turn terminators.
    Llama3,
    /// `[INST]` markup used by Mistral and Llama 2 instruct models.
    MistralInstruct,
}

impl ChatTemplate {
    /// Resolves an explicit template name recorded in a model manifest.
    pub fn from_name(name: &str) -> Option<Self> {
        match name.to_ascii_lowercase().as_str() {
            "chatml" => Some(Self::ChatMl),
            "llama3" => Some(Self::Llama3),
            "mistral" | "llama2" => Some(Self::MistralInstruct),
            _ => None,
        }
    }

    /// Falls back to the conventional template for a registered architecture.
    pub fn for_architecture(architecture: &str) -> Option<Self> {
        match architecture.to_ascii_lowercase().as_str() {
            "qwen3" | "qwen3-gguf" => Some(Self::ChatMl),
            "llama3" => Some(Self::Llama3),
            "llama" | "llama2" | "llama-gguf" | "mistral" | "mixtral" => {
                Some(Self::MistralInstruct)
            }
            _ => None,
        }
    }

    /// Renders the conversation and the opening of the assistant turn.
    pub fn render(&self, messages: &[ChatMessage]) -> Result<String> {
        if messages.is_empty() {
            return Err(EngineError::InvalidGenerationConfig(
                "chat messages must not be empty".to_owned(),
            ));
        }
        match self {
            Self::ChatMl => Ok(render_chatml(messages)),
            Self::Llama3 => Ok(render_llama3(messages)),
            Self::MistralInstruct => render_mistral(messages),
        }
    }
}

fn render_chatml(messages: &[ChatMessage]) -> String {
    let mut prompt = String::new();
    for message in messages {
        prompt.push_str("<|im_start|>");
        prompt.push_str(&message.role);
        prompt.push('\n');
        prompt.push_str(&message.content);
        prompt.push_str("<|im_end|>\n");
    }
    prompt.push_str("<|im_start|>assistant\n");
    prompt
}

fn render_llama3(messages: &[ChatMessage]) -> String {
    let mut prompt = String::from("<|begin_of_text|>");
    for message in messages {
        prompt.push_str("<|start_header_id|>");
        prompt.push_str(&message.role);
        prompt.push_str("<|end_header_id|>\n\n");
        prompt.push_str(&message.content);
        prompt.push_str("<|eot_id|>");
    }
    prompt.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
    prompt
}

fn render_mistral(messages: &[ChatMessage]) -> Result<String> {
    let mut prompt = String::from("<s>");
    let mut pending_system = String::new();
    for message in messages {
        match message.role.as_str() {
            "system" => pending_system = message.content.clone(),
            "user" => {
                let content = if pending_system.is_empty() {
                    message.content.clone()
                } else {
                    format!("{pending_system}\n\n{}", message.content)
                };
                pending_system.clear();
                prompt.push_str("[INST] ");
                prompt.push_str(&content);
                prompt.push_str(" [/INST]");
            }
            "assistant" => {
                prompt.push_str(&message.content);
                prompt.push_str("</s>");
            }
            other => {
                return Err(EngineError::InvalidGenerationConfig(format!(
                    "unsupported chat role for the Mistral template: {other}"
                )));
            }
        }
    }
    Ok(prompt)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn message(role: &str, content: &str) -> ChatMessage {
        ChatMessage {
            role: role.to_owned(),
            content: content.to_owned(),
        }
    }

    #[test]
    fn chatml_renders_system_and_user_turns() {
        let rendered = ChatTemplate::ChatMl
            .render(&[message("system", "You are terse."), message("user", "Hi")])
            .unwrap();
        assert_eq!(
            rendered,
            "<|im_start|>system\nYou are terse.<|im_end|>\n\
             <|im_start|>user\nHi<|im_end|>\n\
             <|im_start|>assistant\n"
        );
    }

    #[test]
    fn llama3_renders_headers_and_turn_terminators() {
        let rendered = ChatTemplate::Llama3
            .render(&[message("user", "Hi")])
            .unwrap();
        assert_eq!(
            rendered,
            "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nHi<|eot_id|>\
             <|start_header_id|>assistant<|end_header_id|>\n\n"
        );
    }

    #[test]
    fn mistral_folds_the_system_prompt_into_the_next_user_turn() {
        let rendered = ChatTemplate::MistralInstruct
            .render(&[
                message("system", "Be brief."),
                message("user", "Hi"),
                message("assistant", "Hello."),
                message("user", "Bye"),
            ])
            .unwrap();
        assert_eq!(
            rendered,
            "<s>[INST] Be brief.\n\nHi [/INST]Hello.</s>[INST] Bye [/INST]"
        );
    }

    #[test]
    fn architecture_fallback_matches_known_families() {
        assert_eq!(
            ChatTemplate::for_architecture("qwen3"),
            Some(ChatTemplate::ChatMl)
        );
        assert_eq!(
            ChatTemplate::for_architecture("llama3"),
            Some(ChatTemplate::Llama3)
        );
        assert_eq!(
            ChatTemplate::for_architecture("mistral"),
            Some(ChatTemplate::MistralInstruct)
        );
        assert_eq!(ChatTemplate::for_architecture("bert"), None);
    }

    #[test]
    fn empty_conversations_are_rejected() {
        assert!(ChatTemplate::ChatMl.render(&[]).is_err());
    }
}
