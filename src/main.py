import threading

from src.gui.app import MainWindow
from src.trainer import Trainer


def main():
    ui = MainWindow()
    trainer = Trainer(
        embedding_size=32,
        ff_dim_size=32,
        max_word_count=16,
        is_train=False,
        temperature=1,
        learning_rate=5e-3,
        train_dataset_len=4,
        on_update=lambda text:
            ui.root.after(0, ui.update, text)
    )

    thread = threading.Thread(
        target=trainer.run,
        daemon=True
    )
    thread.start()

    ui.run()


if __name__ == "__main__":
    main()