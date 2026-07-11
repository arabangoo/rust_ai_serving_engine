use clap::{Args, Parser, Subcommand, ValueEnum};
use rust_ai_serving_engine_api::{ApiState, serve};
use rust_ai_serving_engine_core::{
    DevicePreference, HuggingFaceHub, ModelKind, ModelRegistry, RuntimeDevice,
};
use std::{path::PathBuf, process::ExitCode};

#[derive(Debug, Parser)]
#[command(
    name = "rust-ai-serving-engine",
    version,
    about = "Rust local model runtime registry"
)]
struct Cli {
    #[arg(long, global = true, default_value = ".rust_ai_serving_engine")]
    store: PathBuf,

    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    Model {
        #[command(subcommand)]
        command: ModelCommand,
    },
    Runtime {
        #[command(subcommand)]
        command: RuntimeCommand,
    },
    Serve(ServeArgs),
}

#[derive(Debug, Args)]
struct ServeArgs {
    #[arg(long, default_value = "127.0.0.1")]
    host: String,
    #[arg(long, default_value_t = 8080)]
    port: u16,
    #[arg(long, value_enum, default_value_t = CliDevicePreference::Auto)]
    device: CliDevicePreference,
}

#[derive(Debug, Subcommand)]
enum ModelCommand {
    Import(ImportArgs),
    Pull(PullArgs),
    AttachTokenizer(AttachTokenizerArgs),
    List,
    Inspect { id: String },
    Verify { id: String },
}

#[derive(Debug, Args)]
struct PullArgs {
    #[arg(long)]
    repo: String,
    #[arg(long)]
    file: String,
    #[arg(long)]
    id: String,
    #[arg(long, value_enum, default_value_t = CliModelKind::Generator)]
    kind: CliModelKind,
    #[arg(long)]
    architecture: Option<String>,
    #[arg(long)]
    context_length: Option<u32>,
    #[arg(long)]
    chat_template: Option<String>,
    /// Repository to fetch tokenizer.json from (defaults to --repo when
    /// --tokenizer-file is given).
    #[arg(long)]
    tokenizer_repo: Option<String>,
    /// tokenizer.json artifact to download and attach after registration.
    #[arg(long)]
    tokenizer_file: Option<String>,
}

#[derive(Debug, Args)]
struct AttachTokenizerArgs {
    id: String,
    #[arg(long)]
    tokenizer: PathBuf,
}

#[derive(Debug, Subcommand)]
enum RuntimeCommand {
    Probe {
        #[arg(long, value_enum, default_value_t = CliDevicePreference::Auto)]
        device: CliDevicePreference,
    },
}

#[derive(Debug, Args)]
struct ImportArgs {
    path: PathBuf,
    #[arg(long)]
    id: String,
    #[arg(long, value_enum, default_value_t = CliModelKind::Generator)]
    kind: CliModelKind,
    #[arg(long)]
    architecture: Option<String>,
    #[arg(long)]
    context_length: Option<u32>,
    #[arg(long)]
    chat_template: Option<String>,
}

#[derive(Clone, Debug, ValueEnum)]
enum CliModelKind {
    Generator,
    Embedding,
}

#[derive(Clone, Debug, ValueEnum)]
enum CliDevicePreference {
    Auto,
    Cpu,
    Cuda,
    Metal,
}

impl From<CliModelKind> for ModelKind {
    fn from(value: CliModelKind) -> Self {
        match value {
            CliModelKind::Generator => Self::Generator,
            CliModelKind::Embedding => Self::Embedding,
        }
    }
}

impl From<CliDevicePreference> for DevicePreference {
    fn from(value: CliDevicePreference) -> Self {
        match value {
            CliDevicePreference::Auto => Self::Auto,
            CliDevicePreference::Cpu => Self::Cpu,
            CliDevicePreference::Cuda => Self::Cuda,
            CliDevicePreference::Metal => Self::Metal,
        }
    }
}

#[tokio::main]
async fn main() -> ExitCode {
    match run(Cli::parse()).await {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::FAILURE
        }
    }
}

async fn run(cli: Cli) -> Result<(), Box<dyn std::error::Error>> {
    let registry = ModelRegistry::open(cli.store)?;
    match cli.command {
        Command::Model { command } => match command {
            ModelCommand::Import(args) => {
                let imported = registry.import_local(
                    args.id,
                    args.path,
                    args.kind.into(),
                    args.architecture,
                    args.context_length,
                    args.chat_template,
                )?;
                println!("registered {}", imported.manifest.id);
                println!("manifest: {}", imported.manifest_path.display());
                println!("sha256: {}", imported.manifest.sha256);
            }
            ModelCommand::Pull(args) => {
                let weights = HuggingFaceHub.download(&args.repo, &args.file)?;
                let imported = registry.import_local(
                    &args.id,
                    weights,
                    args.kind.into(),
                    args.architecture,
                    args.context_length,
                    args.chat_template,
                )?;
                println!("downloaded and registered {}", imported.manifest.id);
                println!("manifest: {}", imported.manifest_path.display());
                if let Some(tokenizer_file) = args.tokenizer_file {
                    let tokenizer_repo = args.tokenizer_repo.as_deref().unwrap_or(&args.repo);
                    let tokenizer = HuggingFaceHub.download(tokenizer_repo, &tokenizer_file)?;
                    registry.attach_tokenizer(&args.id, tokenizer)?;
                    println!("attached tokenizer from {tokenizer_repo}/{tokenizer_file}");
                }
            }
            ModelCommand::AttachTokenizer(args) => {
                let manifest = registry.attach_tokenizer(&args.id, &args.tokenizer)?;
                println!("attached tokenizer to {}", manifest.id);
            }
            ModelCommand::List => {
                for model in registry.list()? {
                    println!(
                        "{}\t{:?}\t{:?}\t{}",
                        model.id, model.kind, model.format, model.weights
                    );
                }
            }
            ModelCommand::Inspect { id } => {
                println!("{}", toml::to_string_pretty(&registry.get(&id)?)?);
            }
            ModelCommand::Verify { id } => {
                let model = registry.verify(&id)?;
                println!("verified {} ({})", model.id, model.sha256);
            }
        },
        Command::Runtime { command } => match command {
            RuntimeCommand::Probe { device } => {
                let runtime = RuntimeDevice::select(device.into())?;
                runtime.smoke_test()?;
                println!("backend: {:?}", runtime.kind());
                println!("accelerated: {}", runtime.is_accelerated());
                println!("tensor_smoke_test: passed");
            }
        },
        Command::Serve(args) => {
            let address = format!("{}:{}", args.host, args.port).parse()?;
            println!("serving OpenAI-compatible endpoints on http://{address}");
            serve(
                address,
                ApiState::new(registry.root().to_owned(), args.device.into()),
            )
            .await?;
        }
    }
    Ok(())
}
