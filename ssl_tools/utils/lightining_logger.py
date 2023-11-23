import types
import time


def performance_lightining_logger(cls):
    class Logged(cls):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

            default_time = time.time()
            self.train_epoch_time = default_time
            self.fit_start_time = default_time

        def on_fit_end(self):
            overall_fit_time = time.time() - self.fit_start_time
            print(f"--> Overall fit time: {overall_fit_time:.3f} seconds")
            

        def training_step(self, batch, batch_idx):
            # Start of the step
            start_time = time.time()
            # Perform the step
            result = super().training_step(batch, batch_idx)
            # Calculate the delta time
            delta_time = time.time() - start_time
            # Log the delta time
            self.log(
                "train_step_epoch_time",
                delta_time,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                logger=True,
            )
            return result


        def on_train_epoch_end(self):
            result = super().on_train_epoch_end()
            
            # End of the epoch
            end = time.time()
            # Delta time (this epoch - previous epoch)
            delta_time = end - self.train_epoch_time
            # Set the previous epoch time to the current epoch time
            self.train_epoch_time = end
            # Log it
            self.log(
                "train_epoch_time",
                delta_time,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
            )
            return result

    Logged = types.new_class(cls.__name__, (Logged,), {})
    Logged.__module__ = cls.__module__

    return Logged
