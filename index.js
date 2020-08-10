const PORT = 8080

const express = require('express')
const tf = require('@tensorflow/tfjs')

const app = express()
const http = require('http').createServer(app)
const io = require('socket.io')(http)

const ML = require('./ML')

const ml = new ML()

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

    console.log(`client connected (${socket.request.connection.remoteAddress})`)

    const values = ml.data.map(({ horsepower, mpg }) => ({ x: horsepower, y: mpg }))

    /* socket.emit('scatterplot data', {
      labelX: 'Horsepower',
      labelY: 'Miles per Gallon',
      values: values,
    }) */

    //socket.emit('model summary', ml.model)

    socket.emit('scatterplot results', results)

    socket.on('disconnect', () => {
      console.log(`client disconnected (${socket.request.connection.remoteAddress})`)
    })

  })

  http.listen(PORT, () => console.log(`listening on *:${PORT}`))

}

setTimeout(main, 1)
