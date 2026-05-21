#!/bin/bash
set -e

DOMAIN="phitecia.com"
REPO="https://github.com/alexandredumas88-collab/tradingview-bot.git"
APP_DIR="/opt/tradingview-bot"

echo "==> Updating system packages"
sudo apt update && sudo apt upgrade -y

echo "==> Installing Node.js 20"
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt install -y nodejs

echo "==> Installing PM2"
sudo npm install -g pm2

echo "==> Installing nginx and certbot"
sudo apt install -y nginx certbot python3-certbot-nginx

echo "==> Cloning repository"
sudo git clone $REPO $APP_DIR || (cd $APP_DIR && sudo git pull)
cd $APP_DIR

echo "==> Installing dependencies"
sudo npm install --omit=dev

echo "==> Setting up .env (edit this file before starting the app)"
if [ ! -f .env ]; then
  sudo cp .env.example .env
  echo "  -> Created .env from .env.example — fill in your values now:"
  echo "       sudo nano $APP_DIR/.env"
fi

echo "==> Configuring nginx"
sudo cp nginx.conf /etc/nginx/sites-available/$DOMAIN
sudo ln -sf /etc/nginx/sites-available/$DOMAIN /etc/nginx/sites-enabled/$DOMAIN
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t && sudo systemctl reload nginx

echo "==> Obtaining SSL certificate"
sudo certbot --nginx -d $DOMAIN -d www.$DOMAIN --non-interactive --agree-tos -m admin@$DOMAIN

echo "==> Starting app with PM2"
cd $APP_DIR
sudo pm2 start ecosystem.config.js
sudo pm2 save
sudo pm2 startup systemd -u root --hp /root | tail -1 | bash

echo ""
echo "==> Done! Bot is running at https://$DOMAIN"
echo "    TradingView webhook URL: https://$DOMAIN/webhook"
echo ""
echo "    Useful commands:"
echo "      pm2 logs tradingview-bot   # view logs"
echo "      pm2 restart tradingview-bot # restart"
echo "      pm2 stop tradingview-bot    # stop"
