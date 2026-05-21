const logger = require('../utils/logger');
const { placeOrder, closePosition } = require('../services/broker');

const WEBHOOK_SECRET = process.env.WEBHOOK_SECRET;

async function handleWebhook(req, res) {
  const secret = req.headers['x-webhook-secret'] || req.body?.secret;

  if (!WEBHOOK_SECRET || secret !== WEBHOOK_SECRET) {
    logger.warn('Unauthorized webhook attempt');
    return res.status(401).json({ error: 'Unauthorized' });
  }

  const { symbol, action, volume = 0.01, ticket } = req.body;

  if (!action) {
    return res.status(400).json({ error: 'Missing required field: action' });
  }

  const act = action.toLowerCase();

  if (!['buy', 'sell', 'close'].includes(act)) {
    return res.status(400).json({ error: 'action must be "buy", "sell", or "close"' });
  }

  if (act === 'close') {
    if (!symbol && !ticket) {
      return res.status(400).json({ error: 'close requires symbol or ticket' });
    }
    try {
      const result = await closePosition({ symbol, ticket });
      return res.json({ received: true, close: result });
    } catch (err) {
      logger.error('Failed to close position', err.message);
      return res.status(502).json({ error: 'Failed to close position', detail: err.message });
    }
  }

  if (!symbol) {
    return res.status(400).json({ error: 'Missing required field: symbol' });
  }

  try {
    const result = await placeOrder({ symbol, action, volume });
    res.json({ received: true, order: result });
  } catch (err) {
    logger.error('Failed to place order', err.message);
    res.status(502).json({ error: 'Failed to place order', detail: err.message });
  }
}

module.exports = { handleWebhook };
