# kaustabpal's Blog

Welcome to the source code for my personal blog, hosted at [kaustabpal.github.io](https://kaustabpal.github.io/).

## Overview

This is a static site built with [Hugo](https://gohugo.io/), featuring a custom-designed theme inspired by retro computer terminals and sci-fi aesthetics.

### Key Features
- **Visual Style:** Custom CSS implementing CRT scanlines, screen curvature, vignettes, and text glitch effects.
- **Typography:** Uses *Departure Mono* for headers/UI and *Roboto Mono* for readability.
- **Math Support:** Integrated MathJax for rendering LaTeX mathematical equations.
- **404 Page:** A fully immersive, standalone "signal lost" error page.
- **Automated Deployment:** GitHub Actions workflow automatically builds and deploys to GitHub Pages on every push to `main`.

## Local Development

To run this site locally on your machine:

1.  **Prerequisites:**
    - Install [Hugo](https://gohugo.io/installation/) (Extended version recommended).
    - Install [Git](https://git-scm.com/).

2.  **Clone the repository:**
    ```bash
    git clone https://github.com/kaustabpal/kaustabpal.github.io.git
    cd kaustabpal.github.io
    ```

3.  **Run the development server:**
    ```bash
    hugo server -D
    ```
    The site will be available at `http://localhost:1313/`. The `-D` flag includes draft content.

## Project Structure

- `content/`: Markdown files for blog posts and notes.
- `layouts/`: HTML templates for the site structure.
- `static/`: Static assets like CSS, images, and fonts.
- `.github/workflows/`: CI/CD configuration for GitHub Actions.
- `hugo.toml`: Main site configuration.

## License

This project is dual-licensed:

-   **Source Code**: The underlying code (including templates, CSS, and HTML structure) is licensed under the [MIT License](LICENSE).
-   **Content**: The creative content (blog posts, notes, images) is licensed under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0)](http://creativecommons.org/licenses/by-nc-sa/4.0/).

&copy; Kaustab Pal.
