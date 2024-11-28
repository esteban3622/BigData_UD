# Desafío: investigar el límite de precisión del 90%
## Informe sus hallazgos:
Resuma sus observaciones sobre por qué el modelo está limitado a una precisión del 90%.
Incluya capturas de pantalla o gráficos (por ejemplo, curvas de pérdida, comparaciones de precisión o datos visualizados) para respaldar sus conclusiones.

Las observaciones realizadas se orientan en la construcción del modelo, ya que se considera que este aunque tengo la estructura base de una red LeNet está limitado en su precisión. La red cuenta con el siguiente conjunto y orden de capas:
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
6. Capas densas completamente conectadas x3
```
model.add(tf.layers.dense({ units: 120, activation: 'tanh' }));
model.add(tf.layers.dense({ units: 84, activation: 'tanh' }));
model.add(tf.layers.dense({ units: 10, activation: 'softmax' }));
```
Estas son las caracteristicas con que se compilo el modelo.
```
const model = createLeNetModel();
model.compile({
optimizer: 'sgd',
loss: 'categoricalCrossentropy',
metrics: ['accuracy']
});
```

Así como también como se entreno el modelo.
```
await model.fit(data.trainX, data.trainY, {
  epochs: 5,
  validationData: [data.testX, data.testY],
  callbacks: {
    onEpochEnd: (epoch, logs) => {
      document.getElementById('status').innerText = `Epoch ${epoch + 1}: loss = ${logs.loss.toFixed(4)}, accuracy = ${logs.acc.toFixed(4)}`;
    }
  }
});
```

## Resultados del experimento:
Comparta los cambios que realizó (si los hubo) para mejorar la precisión.
Proporcione métricas de precisión del antes y el después y explique por qué sus ajustes funcionaron (o no funcionaron).


## Proponer soluciones:
Mejora de la implementación actual que podría ayudar a lograr una mayor precisión.

Para ayudar a mejorar la precisión del modelo se propone realizar las siguientes acciones, teniendo en cuenta las caracteristicas de la red:
- De acuerdo con la estructura de la red, se propone la variación de las función de activación de las capas convolucionales, y capas densas.
- De la misma forma es posible empezar a varias la compilación del modelo, como lo es el optimizador de `sgd` a `adam`, aunque se ayuda a una convergencia del resultado en menor epocas.
- Explorar la cantidad de filtros por cada capa convolucional.
- Aumentar la cantidad de epocas para brindar mayor capacidad de extraer patrones de los datos.
- Agregar capas 'dropout' entre las capas convolucionales y densas, puede llegar a permitir que el modelo no sobreaprenda y garatizar resultados diferentes.
- Aumentar la cantidad de capas densas puede llegar a identificar patrones diferentes.
