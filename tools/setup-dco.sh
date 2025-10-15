#!/bin/bash

echo "Setting up DCO for the Cactus project..."

git config core.hooksPath .githooks

echo "✓ Git hooks configured to use .githooks directory"

name=$(git config user.name)
email=$(git config user.email)

if [ -z "$name" ] || [ -z "$email" ]; then
    echo ""
    echo "⚠️  Warning: Git user configuration is incomplete"
    echo ""
    echo "Please configure your git identity:"
    echo "  git config --global user.name \"Your Name\""
    echo "  git config --global user.email \"your.email@example.com\""
    echo ""
else
    echo "✓ Git user configured as: $name <$email>"
fi

echo ""
echo "DCO setup complete!"
echo ""
echo "From now on, your commits will automatically be signed-off."
echo "You can also manually sign commits with: git commit -s"
echo ""
echo "To learn more about contributing, see CONTRIBUTING.md"