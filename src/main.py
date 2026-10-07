import argparse

from trainer import Trainer
from core.config import (
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
        default=TrainConfig.dataset,
        help=f"Training dataset (default: {TrainConfig.dataset})"
    )

    # Evaluate
    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate the model"
    )

    evaluate_parser.add_argument(
        "--dataset",
        default=TrainConfig.dataset,
        help=f"Evaluation dataset (default: {TrainConfig.dataset})"
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
        default=1.0,
        help="Sampling temperature (default: 1.0)"
    )

    generate_parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of top tokens for Top-K sampling (default: 3)"
    )

    generate_parser.add_argument(
        "--top-p",
        type=int,
        default=0.9,
        help="Cumulative probability for Top-P sampling (default: 0.9)"
    )

    args = parser.parse_args()

    model_config = ModelConfig()

    train_config = TrainConfig(
        train_percent=getattr(args, "train_percent", TrainConfig.train_percent),
        epochs=getattr(args, "epochs", TrainConfig.epochs),
        dataset=getattr(args, "dataset", TrainConfig.dataset),
        learning_rate=getattr(args, "learning_rate", TrainConfig.learning_rate)
    )

    generation_config = GenerationConfig(
        temperature=getattr(args, "temperature", 1.0),
        top_k=getattr(args, "top_k", 3),
        top_p=getattr(args, "top_p", 0.9)
    )

    checkpoint_config = CheckpointConfig()

    trainer = Trainer(
        generation_config=generation_config,
        model_config=model_config,
        train_config=train_config,
        checkpoint_config=checkpoint_config
    )

    match args.command:
        case "train":
            trainer.train()

        case "evaluate":
            trainer.evaluate()

        case "generate":
            for i in range(10):
                trainer.generate(args.prompt)


if __name__ == "__main__":
    main()