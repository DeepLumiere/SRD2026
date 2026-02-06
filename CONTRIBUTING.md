# Contributing to SRD2026

Thank you for your interest in contributing to this project! We welcome contributions from the community.

## How to Contribute

### Reporting Issues

If you find a bug or have a feature request, please open an issue on GitHub with:
- A clear description of the problem or suggestion
- Steps to reproduce (for bugs)
- Expected vs actual behavior
- Your environment (OS, Python version, etc.)

### Contributing Code

1. **Fork the repository**
   - Click the "Fork" button on GitHub, or
   - Use the GitHub CLI:
   ```bash
   gh repo fork DeepLumiere/SRD2026 --clone
   ```

2. **Create a new branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes**
   - Follow the existing code style
   - Add tests for new functionality
   - Update documentation as needed

4. **Test your changes**
   ```bash
   pytest tests/
   ```

5. **Commit your changes**
   ```bash
   git commit -m "Description of your changes"
   ```

6. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```

7. **Open a Pull Request**
   - Provide a clear description of your changes
   - Reference any related issues

## Code Style

- Follow PEP 8 guidelines for Python code
- Use meaningful variable and function names
- Add docstrings to all functions and classes
- Keep functions focused and modular

## Testing

- Add unit tests for new features
- Ensure all tests pass before submitting PR
- Aim for high code coverage

## Documentation

- Update README.md if you add new features
- Add docstrings to new functions/classes
- Update configuration examples if needed

## Questions?

Feel free to open an issue for any questions about contributing.

Thank you for contributing to SRD2026!
