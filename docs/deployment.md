# Deployment record

Platform: Render, free web service, deployed from an existing image (no repo access).
Image: ghcr.io/aadityaa-dava/mlops-tabular-pipeline, pinned to a commit-SHA tag.
Port: container listens on 8000; the PORT environment variable on Render must be 8000.
Health check path: /health.

## What went wrong on the way
- Render defaulted PORT to 1000, so its port scan found nothing and the first deploy
  failed after about 5 minutes even though the app had started. Setting PORT=8000 fixed it.

## Measured (rough)
- Warm predict, 2 transactions over the internet: about 1.1 s.
- Startup on Render: about 18 s from service start to the app answering.
- Not yet measured: a cold start after more than 15 idle minutes.

## Why Render
AWS, Google Cloud and Azure require a payment method at signup, and Hugging Face Docker
Spaces now need a paid plan, so Render's free tier was the option that needed no payment
method up front. Whether Render's signup asked for a card has not been confirmed here.

## Updating
Edit the image tag in the service settings (or use Manual Deploy) after CI publishes a new
image. Whether a deploy hook can automate this has not been checked.
