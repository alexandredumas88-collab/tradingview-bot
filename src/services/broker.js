const axios = require('axios');
const logger = require('../utils/logger');

const BRIDGE_URL = process.env.MT5_BRIDGE_URL || 'http://localhost:5000';

async function placeOrder({ symbol, action, volume }) {
  const order = {
    symbol,
    action: action.toUpperCase(),
    volume: parseFloat(volume),
    type: 'MARKET',
    comment: 'TradingView',
  };

  logger.info('Placing order', order);
  const response = await axios.post(`${BRIDGE_URL}/order`, order);
  logger.info('Order response', response.data);
  return response.data;
}

async function closePosition({ symbol, ticket }) {
  const body = ticket ? { ticket } : { symbol };
  logger.info('Closing position', body);
  const response = await axios.post(`${BRIDGE_URL}/close`, body);
  logger.info('Close response', response.data);
  return response.data;
}

module.exports = { placeOrder, closePosition };
