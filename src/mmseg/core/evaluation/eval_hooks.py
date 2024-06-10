import os.path as osp

from mmcv.runner import Hook
from torch.utils.data import DataLoader


class EvalHook(Hook):

    def __init__(self, dataloader, interval=1, by_epoch=False, **eval_kwargs):
        """
        Evaluation hook.

        This hook evaluates the model on a validation dataset at specific intervals during training.

        Args:
            dataloader (DataLoader): A PyTorch DataLoader for the validation dataset.
            interval (int): Evaluation interval in epochs. Default is 1.
            by_epoch (bool): Whether to evaluate by epoch. If False, evaluation is done based on iterations. Default is False.
            **eval_kwargs: Additional keyword arguments passed to the evaluation function.
        """

        # Check if dataloader is a PyTorch DataLoader
        if not isinstance(dataloader, DataLoader):
            raise TypeError('dataloader must be a pytorch DataLoader, but got '
                            f'{type(dataloader)}')

        # Initialize hook attributes
        self.dataloader = dataloader
        self.interval = interval
        self.by_epoch = by_epoch
        self.eval_kwargs = eval_kwargs

    def after_train_iter(self, runner):
        """
        Hook after a training iteration.

        This method is executed after each training iteration. If training is done by epochs or if the evaluation interval of iterations is not reached, the method returns without performing any action. Otherwise, it performs an evaluation of the model on the validation dataset.

        Args:
            runner (Runner): Instance of the training runner.
        """

        # If evaluation is done by epochs or if it's not time to evaluate in this iteration, return without doing
        # anything
        if self.by_epoch or not self.every_n_iters(runner, self.interval):
            return

        # Perform model evaluation on the validation dataset
        from src.mmseg.apis import single_gpu_test
        runner.log_buffer.clear()
        results = single_gpu_test(runner.model, self.dataloader, show=False)
        self.evaluate(runner, results)

    def after_train_epoch(self, runner):
        """
        Hook after a training epoch.

        This method is executed after each training epoch. If evaluation is not done by epochs or if it's not time to evaluate in this epoch, the method returns without performing any action. Otherwise, it performs an evaluation of the model on the validation dataset.

        Args:
            runner (Runner): Instance of the training runner.
        """

        # If evaluation is not done by epochs or if it's not time to evaluate in this epoch, return without doing
        # anything
        if not self.by_epoch or not self.every_n_epochs(runner, self.interval):
            return

        # Perform model evaluation on the validation dataset
        from src.mmseg.apis import single_gpu_test
        runner.log_buffer.clear()
        results = single_gpu_test(runner.model, self.dataloader, show=False)
        self.evaluate(runner, results)

    def evaluate(self, runner, results):
        """
        Calls the dataset evaluation function.

        Args:
            runner (Runner): Instance of the training runner.
            results (dict): Results of the model evaluation on the validation dataset.

        Calls the dataset evaluation function using the results obtained during the model evaluation. Evaluation results are stored in the runner's log buffer for later visualization.
        """

        # Call the dataset evaluation function
        eval_res = self.dataloader.dataset.evaluate(
            results, logger=runner.logger, **self.eval_kwargs)

        # Store the evaluation results in the runner's log buffer
        for name, val in eval_res.items():
            runner.log_buffer.output[name] = val
        runner.log_buffer.ready = True


class DistEvalHook(EvalHook):

    def __init__(self,
                 dataloader,
                 interval=1,
                 gpu_collect=False,
                 by_epoch=False,
                 **eval_kwargs):
        """
        Initializes the distributed evaluation hook.

        Args:
            dataloader (DataLoader): A PyTorch DataLoader.
            interval (int): Evaluation interval (in epochs). Default: 1.
            gpu_collect (bool): Whether to use GPU or CPU to collect results. Default: False.
            by_epoch (bool): Whether evaluation is done per epoch or per iteration. Default: False.
            **eval_kwargs: Additional arguments for evaluation.
        """

        if not isinstance(dataloader, DataLoader):
            raise TypeError(
                'dataloader must be a pytorch DataLoader, but got {}'.format(
                    type(dataloader)))

        self.dataloader = dataloader
        self.interval = interval
        self.gpu_collect = gpu_collect
        self.by_epoch = by_epoch
        self.eval_kwargs = eval_kwargs

    def after_train_iter(self, runner):
        """
        Hook that runs after each training iteration.

        This hook is used to perform model evaluation on the validation dataset
        after each training iteration, if an evaluation interval is specified.

        Args:
            runner: The current `Runner` object managing the training.
        """

        # Checks if evaluation should be performed at this iteration, based on the interval and mode
        if self.by_epoch or not self.every_n_iters(runner, self.interval):
            # If it's not time for evaluation, exits the hook
            return

        # Imports the multi_gpu_test function from the mmseg.apis module to perform evaluation
        from src.mmseg.apis import multi_gpu_test
        # Clears the runner's log buffer
        runner.log_buffer.clear()

        # Performs evaluation on multiple GPUs The multi_gpu_test method is passed the model, the validation
        # dataloader, the temporary folder to store results, and the flag to collect results on GPU
        results = multi_gpu_test(
            runner.model,
            self.dataloader,
            tmpdir=osp.join(runner.work_dir, '.eval_hook'),
            gpu_collect=self.gpu_collect)

        # If it's the process with rank 0 (usually the master process), prints a newline
        # and calls the evaluate method to process evaluation results
        if runner.rank == 0:
            print('\n')
            self.evaluate(runner, results)

    def after_train_epoch(self, runner):
        """
        Hook that runs after each training epoch.

        This hook is used to perform model evaluation on the validation dataset
        after each training epoch, if an evaluation interval is specified.

        Args:
            runner: The current `Runner` object managing the training.
        """

        # Checks if evaluation should be performed at this epoch, based on the interval and mode
        if not self.by_epoch or not self.every_n_epochs(runner, self.interval):
            # If it's not time for evaluation, exits the hook
            return

        # Imports the multi_gpu_test function from the mmseg.apis module to perform evaluation
        from src.mmseg.apis import multi_gpu_test
        # Clears the runner's log buffer
        runner.log_buffer.clear()

        # Performs evaluation on multiple GPUs
        # The multi_gpu_test method is passed the model, the validation dataloader, the temporary folder
        # to store results, and the flag to collect results on GPU.
        results = multi_gpu_test(
            runner.model,
            self.dataloader,
            tmpdir=osp.join(runner.work_dir, '.eval_hook'),
            gpu_collect=self.gpu_collect)

        # If it's the process with rank 0 (usually the master process), prints a newline
        # and calls the evaluate method to process evaluation results.
        if runner.rank == 0:
            print('\n')
            self.evaluate(runner, results)
