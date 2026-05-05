# Storifai — Improved

AI-powered visual storytelling using CLIP ViT-B/32 + Cross-Image Attention + Transformer Decoder.

This is the improved version of Storifai, built for CS 747 Deep Learning at George Mason University. It generates 3 differently-toned story versions (Warm, Vivid, Reflective) from 3 to 5 user-uploaded photos, allowing users to select the version that best fits their memory.

## Architecture

- Image Encoder: CLIP ViT-B/32 (replaces ResNet-50 in baseline)
- Cross-Image Context: Multi-head Attention (new in improved)
- Text Decoder: Transformer Decoder (replaces LSTM)
- Story Output: 3 versions per upload (Warm, Vivid, Reflective)

## Project Structure

- api.py — FastAPI backend
- model.py — Architecture definition
- train.py — Training pipeline
- Dockerfile — Cloud Run container
- website/index.html — Frontend (deployed to Vercel)

## Dataset

Trained on VIST (Visual Storytelling) — 40,155 human-written photo stories.

## Author

Ahmad Nesar Sediqzada — Hoshmand AI
