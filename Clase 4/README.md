# Desafío: investigar el límite de precisión del 90%
## Informe sus hallazgos:
Resuma sus observaciones sobre por qué el modelo está limitado a una precisión del 90%.

Para describir posibles razones por las que es modelo está limitado en su precisión es necesario tener en cuenta su configuración, la cual iniciamos. La red convolucional cuenta con las siguientes capas y esta construida de la siguiente manera:
1. Primera capa convolucional
```
model.add(tf.layers.conv2d({
  inputShape: [28, 28, 1],
  filters: 6,
  kernelSize: 5,
  activation: 'tanh',
  padding: 'same'
}));
```
2. Primera capa de pooling
```
model.add(tf.layers.averagePooling2d({ poolSize: 2, strides: 2 }));
```
3. Segunda capa convolucional
```
model.add(tf.layers.conv2d({
  filters: 16,
  kernelSize: 5,
  activation: 'tanh'
}));
```
4. Segunda capa de pooling
```
model.add(tf.layers.averagePooling2d({ poolSize: 2, strides: 2 }));
```
5. Capa de aplanamiento
```
// Flatten layer
model.add(tf.layers.flatten());
```
6. Capas densas completamente conectadas
```
model.add(tf.layers.dense({ units: 120, activation: 'tanh' }));
model.add(tf.layers.dense({ units: 84, activation: 'tanh' }));
model.add(tf.layers.dense({ units: 10, activation: 'softmax' }));
```
Es importante señalar que 
```
const model = createLeNetModel();
model.compile({
optimizer: 'sgd',
loss: 'categoricalCrossentropy',
metrics: ['accuracy']
});
```

Teniendo en cuenta las caracteristicas de la red puede se postulan algunas hipotesis para variar el modelo para aumentar su rendimiento:
- De acuerdo con la estructura de la red, se propone la variación de las función de activación de las capas convolucionales, y capas densas.
- De la misma forma es posible empezar a varias la compilación del modelo, como lo es el optimizador de sgd a adam, aunque se ayuda a una convergencia del resultado en menor epocas.
- Agregar capas 'dropout' puede llegar a permitir que el modelo no sobreaprenda y garatizar resultados diferentes.

Incluya capturas de pantalla o gráficos (por ejemplo, curvas de pérdida, comparaciones de precisión o datos visualizados) para respaldar sus conclusiones.


## Resultados del experimento:
Comparta los cambios que realizó (si los hubo) para mejorar la precisión.
Proporcione métricas de precisión del antes y el después y explique por qué sus ajustes funcionaron (o no funcionaron).
## Proponer soluciones:
Mejora de la implementación actual que podría ayudar a lograr una mayor precisión.
