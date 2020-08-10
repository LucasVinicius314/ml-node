const tf = require('@tensorflow/tfjs')
const fetch = require('node-fetch')

module.exports = class ML {

  // methods

  async GetData() {
    const carsDataReq = await fetch('https://storage.googleapis.com/tfjs-tutorials/carsData.json')
    const carsData = await carsDataReq.json()
    const cleaned = carsData.map(car => ({
      mpg: car.Miles_per_Gallon,
      horsepower: car.Horsepower,
    }))
      .filter(car => (car.mpg != null && car.horsepower != null))

    this.data = cleaned
  }

  CreateModel() {
    const model = tf.sequential()
    model.add(tf.layers.dense({ inputShape: [1], units: 1, useBias: true }))
    model.add(tf.layers.dense({ units: 1, useBias: true }))

    this.model = model
  }

  ConvertToTensor() {

    const data = this.data

    this.tensorData = tf.tidy(() => {

      tf.util.shuffle(data)

      // Convert data to Tensor
      const inputs = data.map(d => d.horsepower)
      const labels = data.map(d => d.mpg)

      const inputTensor = tf.tensor2d(inputs, [inputs.length, 1])
      const labelTensor = tf.tensor2d(labels, [labels.length, 1])

      // Normalize the data
      const inputMax = inputTensor.max()
      const inputMin = inputTensor.min()
      const labelMax = labelTensor.max()
      const labelMin = labelTensor.min()

      const normalizedInputs = inputTensor.sub(inputMin).div(inputMax.sub(inputMin))
      const normalizedLabels = labelTensor.sub(labelMin).div(labelMax.sub(labelMin))

      return {
        inputs: normalizedInputs,
        labels: normalizedLabels,
        // Return the min/max bounds so we can use them later.
        inputMax,
        inputMin,
        labelMax,
        labelMin,
      }
    })
  }

  async TrainModel() {

    const { inputs, labels } = this.tensorData

    // Prepare the model for training
    this.model.compile({
      optimizer: 'sgd',
      loss: tf.losses.meanSquaredError,
      metrics: ['mse'],
    })

    const batchSize = 64
    const epochs = 1000

    return await this.model.fit(inputs, labels, {
      batchSize,
      epochs,
      shuffle: true,
      callbacks: {
        onEpochEnd: (epoch, log) => console.log(`Epoch ${epoch}: loss = ${log.loss}`),
      },
    })
  }

  TestModel() {
    const { inputMax, inputMin, labelMin, labelMax } = this.tensorData

    // Generate predictions for a uniform range of numbers between 0 and 1;
    // We un-normalize the data by doing the inverse of the min-max scaling 
    // that we did earlier.
    const [xs, preds] = tf.tidy(() => {

      const xs = tf.linspace(0, 1, 100)
      const preds = this.model.predict(xs.reshape([100, 1]))

      const unNormXs = xs
        .mul(inputMax.sub(inputMin))
        .add(inputMin)

      const unNormPreds = preds
        .mul(labelMax.sub(labelMin))
        .add(labelMin)

      // Un-normalize the data
      return [unNormXs.dataSync(), unNormPreds.dataSync()]
    })

    const predictedPoints = Array.from(xs).map((val, i) => {
      return { x: val, y: preds[i] }
    })

    const originalPoints = this.data.map(d => ({
      x: d.horsepower, y: d.mpg,
    }))

    return [originalPoints, predictedPoints]
  }

  async BakeModel() {
    const model = tf.sequential()
    model.add(tf.layers.dense({ units: 100, activation: 'relu', inputShape: [10] }))
    model.add(tf.layers.dense({ units: 1, activation: 'linear' }))
    model.compile({ optimizer: 'sgd', loss: 'meanSquaredError' })

    const xs = tf.randomNormal([100, 10])
    const ys = tf.randomNormal([100, 1])

    this.model = model

    model.fit(xs, ys, {
      epochs: 100,
      callbacks: {
        onEpochEnd: (epoch, log) => console.log(`Epoch ${epoch}: loss = ${log.loss}`)
      }
    })
  }

}
