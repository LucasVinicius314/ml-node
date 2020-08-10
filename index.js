const PORT = 8080

const express = require('express')
const tf = require('@tensorflow/tfjs')

const app = express()
const http = require('http').createServer(app)
const io = require('socket.io')(http)

app.use(express.static('public'))

const ML = require('./ML')
const util = require('./util')

const ml = new ML()

ml.optimizer = 'sgd'
ml.batchSize = 64
ml.epochs = 310

const main = async () => {

  // body

  await ml.GetData()
  //console.log(ml.data)

  ml.CreateModel()
  //console.log(ml.model)

  ml.ConvertToTensor()
  //console.log(ml.tensorData)

  await ml.TrainModel()
  //console.log('finished training')

  const results = ml.TestModel()

  // routes

  app.get('/', (req, res) => {
    res.sendFile(__dirname + '/index.html')
  })

  io.on('connection', async socket => {

    console.log(`${util.now()} client connected (${socket.request.connection.remoteAddress})`)

    const values = ml.data.map(({ horsepower, mpg }) => ({ x: horsepower, y: mpg }))
    /* socket.emit('scatterplot data', {
      labelX: 'Horsepower',
      labelY: 'Miles per Gallon',
      values: values,
    }) */

    //socket.emit('model summary', ml.model)

    const loss = Math.floor(ml.loss * 10000) / 100
    socket.emit('scatterplot results', { results: results, accuracy: 100 - loss, epochs: ml.epochs })

    socket.on('disconnect', () => {
      console.log(`${util.now()} client disconnected (${socket.request.connection.remoteAddress})`)
    })

  })

  http.listen(PORT, () => console.log(`listening on *:${PORT}`))

}

setTimeout(main, 1)
