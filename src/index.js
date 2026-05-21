require('dotenv').config();
const express = require('express');
const { handleWebhook } = require('./handlers/webhook');
const logger = require('./utils/logger');

const app = express();
app.use(express.json());

app.post('/webhook', handleWebhook);

app.get('/health', (req, res) => res.json({ status: 'ok' }));

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => logger.info(`Server running on port ${PORT}`));
