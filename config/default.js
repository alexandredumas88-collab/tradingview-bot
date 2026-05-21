module.exports = {
  port: process.env.PORT || 3000,
  webhookSecret: process.env.WEBHOOK_SECRET || '',
  broker: {
    apiKey: process.env.BROKER_API_KEY || '',
    apiSecret: process.env.BROKER_API_SECRET || '',
  },
};
