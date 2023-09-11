# Machine Learning Implementation with framework (Tensorflow)

<h2>Diego Alfonso Ramírez Montes - A01707596</h2>

<p>En esta actividad se solicitó realizar una implementación de Machine Learning con Tensorflow para un data set que podía ser escogido a decisión del alumno.</p>
<p>El data set que se escogió para esta actividad fue "DryBeanDataset" el cual contine 16 caracteristicas de frijoles y su respectiva clase.</p>
<p>El data set no contiene instancias faltantes en ninguno de los espacios descritos, por lo que se podía trabajar directamente con el mismo.</p>
<p>Para el desarrollo del código se optó por utilizar Tensorflow, pues ya se ha trabajado con el con anterioridad (A diferencia de Pytorch).</p>
<p>Se hizo uso de la libreria de PANDAS a fin de realizar One-Hot encoding a las clases a predecir.</p>
<p>Se realizó el split de los datos en training y test por medio de una función de SKLEARN.</p>
<p>Nuevamente se hizo uso de de la libreria SKLEARN para estandarizar los datos.</p>
<p>Se definió el modelo a usar, donde se definió el uso de 64 neuronas para la 1er capa oculta y 32 para la 2da, en ambas se utilizó la función de activación sigmoide. Para la capa de salida se definieron las 7 clases y se utilizó softmax como activación para realizar las predicciones.</p>
<p>Se compiló el modelo, donde se definió un parámetro de optimización, así como la métrica de pérdida y aquella que se debe rastrear (Accuracy).</p>
<p>Se entrenó el modelo en el cual se definieron 100 epochs.</p>
<p>Se evaluó el modelo por medio de la función "model.evaluate()".</p>
<p>Se construyó la matriz de confusión.</p>
<p>Se extrajeron los valores últimos de "loss", "accuracy" de train y "accuracy" de test.</p>
<p>Se realizó el plot de los resultados.</p>
<p>Una vez más se pide disculpas dado que la matriz de confusión no tiene etiquetas.</p>
<p>Se calcularos métricas que ayudan a entender de mejor forma la precisión del modelo:</p>
<p>‌     ‌"accuracy": Cuantas veces acertó nuestro modelo.</p>
<p>‌     ‌"precision": De las veces en que predijo una clase, cuantas fueron correctas.</p>
<p>‌     ‌"recall": Mide cuántas veces el modelo identificó correctamente todos los casos relevantes de una clase.</p>
<p>‌     ‌"f1": Es la media armónica de 'precision' y 'recall'. Nos ayuda a evaluar la eficacia general de nuestro modelo de clasificación.</p>
