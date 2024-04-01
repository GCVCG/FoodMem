import os.path as osp

from mmcv.runner import Hook
from torch.utils.data import DataLoader


class EvalHook(Hook):
    """
    Gancho de evaluación.

    Este gancho evalúa el modelo en un conjunto de datos de validación en intervalos específicos durante el entrenamiento.

    Args:
        dataloader (DataLoader): Un DataLoader de PyTorch para el conjunto de datos de validación.
        interval (int): Intervalo de evaluación en épocas. El valor predeterminado es 1.
        by_epoch (bool): Si se evalúa por época. Si es False, la evaluación se realiza en base a las iteraciones. El valor predeterminado es False.
        **eval_kwargs: Argumentos de palabras clave adicionales pasados a la función de evaluación.

    Attributes:
        dataloader (DataLoader): El DataLoader para el conjunto de datos de validación.
        interval (int): Intervalo de evaluación en épocas.
        by_epoch (bool): Si se evalúa por época.
        eval_kwargs: Argumentos de palabras clave adicionales para la evaluación.
    """

    def __init__(self, dataloader, interval=1, by_epoch=False, **eval_kwargs):
        """
        Gancho de evaluación.

        Este gancho evalúa el modelo en un conjunto de datos de validación en intervalos específicos durante el entrenamiento.

        Args:
            dataloader (DataLoader): Un DataLoader de PyTorch para el conjunto de datos de validación.
            interval (int): Intervalo de evaluación en épocas. El valor predeterminado es 1.
            by_epoch (bool): Si se evalúa por época. Si es False, la evaluación se realiza en base a las iteraciones. El valor predeterminado es False.
            **eval_kwargs: Argumentos de palabras clave adicionales pasados a la función de evaluación.
        """
        # Verifica que el dataloader sea un DataLoader de PyTorch
        if not isinstance(dataloader, DataLoader):
            raise TypeError('dataloader must be a pytorch DataLoader, but got '
                            f'{type(dataloader)}')
        # Inicializa los atributos del gancho
        self.dataloader = dataloader
        self.interval = interval
        self.by_epoch = by_epoch
        self.eval_kwargs = eval_kwargs

    def after_train_iter(self, runner):
        """
        Gancho después de una iteración de entrenamiento.

        Este método se ejecuta después de cada iteración de entrenamiento. Si el entrenamiento se realiza por épocas o si no se alcanza el intervalo de evaluación de iteraciones, el método retorna sin realizar ninguna acción. De lo contrario, realiza una evaluación del modelo en el conjunto de datos de validación.

        Args:
            runner (Runner): Instancia del corredor de entrenamiento.
        """
        # Si se realiza la evaluación por épocas o si no es tiempo de evaluar en esta iteración, retorna sin hacer nada
        if self.by_epoch or not self.every_n_iters(runner, self.interval):
            return
        # Realiza la evaluación del modelo en el conjunto de datos de validación
        from src.mmseg.apis import single_gpu_test
        runner.log_buffer.clear()
        results = single_gpu_test(runner.model, self.dataloader, show=False)
        self.evaluate(runner, results)

    def after_train_epoch(self, runner):
        """
        Gancho después de una época de entrenamiento.

        Este método se ejecuta después de cada época de entrenamiento. Si la evaluación no se realiza por épocas o si no es tiempo de evaluar en esta época, el método retorna sin realizar ninguna acción. De lo contrario, realiza una evaluación del modelo en el conjunto de datos de validación.

        Args:
            runner (Runner): Instancia del corredor de entrenamiento.
        """
        # Si la evaluación no se realiza por épocas o si no es tiempo de evaluar en esta época, retorna sin hacer nada
        if not self.by_epoch or not self.every_n_epochs(runner, self.interval):
            return
        # Realiza la evaluación del modelo en el conjunto de datos de validación
        from src.mmseg.apis import single_gpu_test
        runner.log_buffer.clear()
        results = single_gpu_test(runner.model, self.dataloader, show=False)
        self.evaluate(runner, results)

    def evaluate(self, runner, results):
        """
        Llama a la función de evaluación del conjunto de datos.

        Args:
            runner (Runner): Instancia del corredor de entrenamiento.
            results (dict): Resultados de la evaluación del modelo en el conjunto de datos de validación.

        Llama a la función de evaluación del conjunto de datos de validación utilizando los resultados obtenidos durante la evaluación del modelo. Los resultados de la evaluación se almacenan en el buffer de registro del corredor para su posterior visualización.
        """
        # Llama a la función de evaluación del conjunto de datos de validación
        eval_res = self.dataloader.dataset.evaluate(
            results, logger=runner.logger, **self.eval_kwargs)
        # Almacena los resultados de la evaluación en el buffer de registro del corredor
        for name, val in eval_res.items():
            runner.log_buffer.output[name] = val
        runner.log_buffer.ready = True


class DistEvalHook(EvalHook):
    """
    Gancho de evaluación distribuida.

    Atributos:
        dataloader (DataLoader): Un DataLoader de PyTorch.
        interval (int): Intervalo de evaluación (por épocas). Por defecto: 1.
        tmpdir (str | None): Directorio temporal para guardar los resultados de todos los
            procesos. Por defecto: None.
        gpu_collect (bool): Si se debe utilizar GPU o CPU para recopilar los resultados.
            Por defecto: False.
    """

    def __init__(self,
                 dataloader,
                 interval=1,
                 gpu_collect=False,
                 by_epoch=False,
                 **eval_kwargs):
        """
        Inicializa el gancho de evaluación distribuida.

        Args:
            dataloader (DataLoader): Un DataLoader de PyTorch.
            interval (int): Intervalo de evaluación (por épocas). Por defecto: 1.
            gpu_collect (bool): Si se debe utilizar GPU o CPU para recopilar los resultados.
                Por defecto: False.
            by_epoch (bool): Si la evaluación se realiza por época o por iteración. 
                Por defecto: False.
            **eval_kwargs: Argumentos adicionales para la evaluación.
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
        Gancho que se ejecuta después de cada iteración de entrenamiento.

        Este gancho se utiliza para realizar la evaluación del modelo en el conjunto de datos de evaluación
        después de cada iteración de entrenamiento, si se ha especificado un intervalo para la evaluación.

        Args:
            runner: El objeto `Runner` actual que gestiona el entrenamiento.
        """
        # Verifica si se debe realizar la evaluación en esta iteración, según el intervalo y el modo
        if self.by_epoch or not self.every_n_iters(runner, self.interval):
            # Si no es el momento de la evaluación, sale del gancho
            return
        # Importa la función multi_gpu_test del módulo mmseg.apis para realizar la evaluación
        from src.mmseg.apis import multi_gpu_test
        # Limpia el buffer de registro del runner
        runner.log_buffer.clear()
        # Realiza la evaluación en múltiples GPU
        # Se pasan al método multi_gpu_test el modelo, el dataloader de evaluación, la carpeta temporal para almacenar los resultados 
        # y el indicador para colectar resultados en GPU
        results = multi_gpu_test(
            runner.model,
            self.dataloader,
            tmpdir=osp.join(runner.work_dir, '.eval_hook'),
            gpu_collect=self.gpu_collect)
        # Si es el proceso con rango 0 (por lo general el proceso principal), se imprime un salto de línea
        # y se llama al método evaluate para procesar los resultados de la evaluación
        if runner.rank == 0:
            print('\n')
            self.evaluate(runner, results)

    def after_train_epoch(self, runner):
        """
        Gancho que se ejecuta después de cada época de entrenamiento.

        Este gancho se utiliza para realizar la evaluación del modelo en el conjunto de datos de evaluación
        después de cada época de entrenamiento, si se ha especificado un intervalo para la evaluación.

        Args:
            runner: El objeto `Runner` actual que gestiona el entrenamiento.
        """
        # Verifica si se debe realizar la evaluación en esta iteración, según el intervalo y el modo
        if not self.by_epoch or not self.every_n_epochs(runner, self.interval):
            # Si no es el momento de la evaluación, sale del gancho
            return
        # Importa la función multi_gpu_test del módulo mmseg.apis para realizar la evaluación
        from src.mmseg.apis import multi_gpu_test
        # Limpia el buffer de registro del runner
        runner.log_buffer.clear()
        # Realiza la evaluación en múltiples GPU.
        # Se pasan al método multi_gpu_test el modelo, el dataloader de evaluación, la carpeta temporal
        # para almacenar los resultados y el indicador para colectar resultados en GPU.
        results = multi_gpu_test(
            runner.model,
            self.dataloader,
            tmpdir=osp.join(runner.work_dir, '.eval_hook'),
            gpu_collect=self.gpu_collect)
        # Si es el proceso con rango 0 (por lo general el proceso principal), se imprime un salto de línea
        # y se llama al método evaluate para procesar los resultados de la evaluación.
        if runner.rank == 0:
            print('\n')
            self.evaluate(runner, results)
