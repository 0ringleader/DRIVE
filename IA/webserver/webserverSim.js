const express = require('express');
const app = express();
const http = require('http').Server(app);
const io = require('socket.io')(http);

app.use(express.static('public'));

let speed = 0;
let steering = 0;

io.on('connection', (socket) => {
    socket.emit('speed', speed);
    socket.emit('steering', steering);

    socket.on('speed', (newSpeed) => {
        speed = newSpeed;
        io.emit('speed', speed);
    });

    socket.on('steering', (newSteering) => {
        steering = newSteering;
        io.emit('steering', steering);
    });

    socket.on('keydown', (key) => {
        if (key === 'ArrowUp') {
            speed = Math.min(speed + 0.02, 1);
            io.emit('speed', speed);
        }
        if (key === 'ArrowDown') {
            speed = Math.max(speed - 0.02, 0);
            io.emit('speed', speed);
        }
        if (key === 'ArrowLeft') {
            steering = Math.max(steering - 0.1, -1);
            io.emit('steering', steering);
        }
        if (key === 'ArrowRight') {
            steering = Math.min(steering + 0.1, 1);
            io.emit('steering', steering);
        }
    });
});

http.listen(3000, () => {
    console.log('listening on *:3000');
});

app.get('/', (req, res) => {
    res.sendFile(__dirname + '/index.html');
});

app.get('/values', (req, res) => {
    res.json({ speed, steering });
});