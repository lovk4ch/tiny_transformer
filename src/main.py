import threading

from src.core.types.mode import Mode
from src.gui.app import MainWindow
from src.trainer import Trainer, SamplingMethod


def main():
    # ui = MainWindow()
    trainer = Trainer(
        mode=Mode.EVALUATE,
        embedding_size=32,
        ff_dim_size=32,
        max_tokens=64,
        temperature=1,
        epochs=45,
        learning_rate=5e-3,
        train_dataset_len=0,
        sampling=SamplingMethod.TOP_K,
        # on_update=lambda text:
        #     ui.root.after(0, ui.update, text)
    )

    thread = threading.Thread(
        target=trainer.run,
        daemon=False
    )
    thread.start()

    # ui.run()


if __name__ == "__main__":
    main()