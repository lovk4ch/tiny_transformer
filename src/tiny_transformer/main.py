import argparse

from tiny_transformer.trainer import Trainer
from tiny_transformer.core.config import (
    ModelConfig,
    TrainConfig,
    GenerationConfig, CheckpointConfig
)


VERSION = "0.1.0"

def percent(value):
    value = float(value)

    if not 0 <= value <= 100:
        raise argparse.ArgumentTypeError(
            "must be between 0 and 100"
        )

    return value

def main():
    parser = argparse.ArgumentParser(
        description="Tiny Transformer language model"
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {VERSION}"
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True
    )

    # Train
    train_parser = subparsers.add_parser(
        "train",
        help="Train the model"
    )

    train_parser.add_argument(
        "--train-percent",
        type=percent,
        default=TrainConfig.train_percent,
        help=f"Training dataset percentage (default: {TrainConfig.train_percent})"
    )

    train_parser.add_argument(
        "--epochs",
        type=int,
        default=TrainConfig.epochs,
        help=f"Number of training epochs (default: {TrainConfig.epochs})"
    )

    train_parser.add_argument(
        "--learning_rate",
        default=TrainConfig.learning_rate,
        help=f"Learning rate (default: {TrainConfig.learning_rate})"
    )

    train_parser.add_argument(
        "--dataset",
        default=TrainConfig.dataset_path,
        help=f"Training dataset (default: {TrainConfig.dataset_path})"
    )

    # Evaluate
    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate the model"
    )

    evaluate_parser.add_argument(
        "--dataset",
        default=TrainConfig.dataset_path,
        help=f"Evaluation dataset (default: {TrainConfig.dataset_path})"
    )

    # Generate
    generate_parser = subparsers.add_parser(
        "generate",
        help="Generate text"
    )

    generate_parser.add_argument(
        "prompt",
        help="Starting text for generation"
    )

    generate_parser.add_argument(
        "--temperature",
        type=float,
        default=GenerationConfig.temperature,
        help="Sampling temperature (default: 1.0)"
    )

    generate_parser.add_argument(
        "--top-k",
        type=int,
        default=GenerationConfig.top_k,
        help="Number of top tokens for Top-K sampling (default: 3)"
    )

    generate_parser.add_argument(
        "--top-p",
        type=int,
        default=GenerationConfig.top_p,
        help="Cumulative probability for Top-P sampling (default: 0.9)"
    )

    args = parser.parse_args()

    generation_config = GenerationConfig(
        temperature=getattr(args, "temperature", GenerationConfig.temperature),
        top_k=getattr(args, "top_k", GenerationConfig.top_k),
        top_p=getattr(args, "top_p", GenerationConfig.top_p)
    )

    model_config = ModelConfig()
    checkpoint_config = CheckpointConfig()

    train_config = TrainConfig(
        train_percent=getattr(args, "train_percent", TrainConfig.train_percent),
        epochs=getattr(args, "epochs", TrainConfig.epochs),
        dataset_path=getattr(args, "dataset", TrainConfig.dataset_path),
        learning_rate=getattr(args, "learning_rate", TrainConfig.learning_rate)
    )

    trainer = Trainer(
        generation_config=generation_config,
        model_config=model_config,
        checkpoint_config=checkpoint_config,
        train_config=train_config
    )

    match args.command:
        case "train":
            trainer.train()

        case "evaluate":
            trainer.evaluate()

        case "generate":
            for i in range(1):
                trainer.generate(args.prompt)


if __name__ == "__main__":
    main()