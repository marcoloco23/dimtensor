# Physics with dimtensor: An Interactive Textbook

Welcome to **Physics with dimtensor**, an interactive textbook that teaches physics through unit-aware computation.

## What You'll Learn

This textbook will teach you:

- **Physical Units & Dimensions**: Understand the fundamental building blocks of physics
- **Dimensional Analysis**: Use dimensions to derive equations and catch errors
- **Unit-Aware Computing**: Write physics code that automatically tracks units
- **Problem Solving**: Apply computational physics to real-world problems

## How to Use This Book

Each chapter contains:

- **Conceptual Lessons**: Clear explanations of physics concepts
- **Worked Examples**: Step-by-step problem solutions
- **Interactive Exercises**: Practice problems with instant feedback
- **Code Examples**: Python code using dimtensor

## Getting Started

### Installation

First, install dimtensor:

```bash
pip install dimtensor[education]
```

### Your First DimArray

```python
from dimtensor import DimArray, units

# Create a velocity
velocity = DimArray([10, 20, 30], units.m / units.s)
print(velocity)  # [10 20 30] m/s

# Create a time
time = DimArray([1, 2, 3], units.s)

# Calculate distance
distance = velocity * time
print(distance)  # [10 40 90] m
```

## Chapters

1. **Units and Measurements** - Learn about physical units and the SI system
2. **Dimensional Analysis** - Master the art of dimensional reasoning
3. **Basic Operations** - Perform calculations while tracking units

More chapters coming soon!

## Features

- 🎓 **Auto-Grading**: Get instant feedback on your answers
- 📊 **Progress Tracking**: Monitor your learning progress
- 💡 **Hints**: Get help when you're stuck
- 🧪 **Interactive**: Run code directly in the notebook

Let's get started with [Chapter 1: Units and Measurements](intro/chapter_01_units.ipynb)!
