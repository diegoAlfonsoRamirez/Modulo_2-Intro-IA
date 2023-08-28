# Machine Learning Implementation

<h2>Diego Alfonso Ramírez Montes - A01707596</h2>

<p style="text-align:justify;">En esta actividad se solicitó realizar una implementación de Machine Learning para un data set que podía ser escogido a decisión del alumno.</p>
<p>El data set que se escogió para esta actividad fue "DryBeanDataset" el cual contine 16 caracteristicas de frijoles y su respectiva clase.</p>
<p>El data set no contiene instancias faltantes en ninguno de los espacios descritos, por lo que se podía trabajar directamente con el mismo.</p>
<p>Para el desarrollo del código se optó por una regresión lógistica al ser la más adecuada al lidiar con una elección entre varias clases.</p>
<p>Se optó a su vez por el uso de la técinica de "One-Hot encoding" pues las etiquetas de clase son cadenas de texto (strings).</p>
<p>El uso de las librerias de pandas y numpy fue necesario a fin de manipular los datos de forma más eficiente, así mismo matplotlib.pyplot fue usada para graficar variables de interés.</p>
<p>El uso de cualquier otra libreria fue restringido.</p>
<p>El data set fue dividido en 80% para entrenamiento de la red y el 20% para la prueba de la misma, se decidió utilizar estos valores dado el tamño del data set (13, 611 datos).</p>
<p>También se añadió una matriz de confusión a fin de poder visualizar las clasificaciones finales del modelo, se esperaría una linea diagonal con aproximadamente un séptimo del total de datos, pues hay 7 clases.</p>
<p>La matriz de confusión no despliega los nombres de las clases de la tabla, se pide una disculpa por lo mismo.</p>
