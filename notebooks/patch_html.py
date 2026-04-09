import os
import re

def fix_jupyterbook_html(input_dir, output_dir):
    """
    Fix all HTML files in input_dir to:
    1. Remove window.MathJax script (if present)
    2. Remove CDN MathJax script (if present)
    3. Add local MathJax script: mathjax/MathJax.js?config=TeX-AMS-MML_SVG
    4. Add CSS style for body font
    5. Remove 'tex2jax_ignore' and 'mathjax_ignore' classes from any section/div
    """
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    for root, _, files in os.walk(input_dir):
        for file_path in files:
            if not file_path.endswith('.html'):
                continue
                
            input_file = os.path.join(root, file_path)
            relative_path = os.path.relpath(file_path, '.')
            output_file = os.path.join(output_dir, relative_path)

            with open(input_file, 'r', encoding='utf-8') as f:
                html_content = f.read()

            # 🔁 1. Remove window.MathJax script (if exists)
            html_content = re.sub(
                r'<script\s+[^>]*window\.MathJax\s*=\s*{[^}]+}</script>',
                '', 
                html_content, 
                flags=re.IGNORECASE
            )

            # 🔁 2. Remove CDN MathJax script (if exists)
            html_content = re.sub(
                r'<script[^>]*src="https://cdn\.jsdelivr\.net/npm/mathjax@3/es5/tex-mml-chtml\.js"[^>]*></script>',
                '', 
                html_content, 
                flags=re.IGNORECASE
            )

            # 🔁 3. Add local MathJax script (inside <head>)
            mathjax_script = '''
<script type="text/javascript" src="mathjax/MathJax.js?config=TeX-AMS-MML_SVG"></script>
'''
            
            # Find position to insert before </head> or at top of head
            head_end = html_content.find('</head>')
            if head_end == -1:
                print(f"⚠️ Missing </head> in {file_path} — cannot inject MathJax")
                continue

            # Insert script BEFORE closing </head>
            new_html = html_content[:head_end] + mathjax_script + html_content[head_end:]

            # 🔁 4. Add CSS style (body font)
            css_style = '''
<style>
body { font-family: Arial, sans-serif, Asana-Math; }
</style>
'''
            
            # Find end of <head> to insert CSS
            head_end_2 = new_html.find('</head>')
            if head_end_2 == -1:
                print(f"⚠️ Missing </head> in {file_path} — cannot inject style")
                continue

            new_html = new_html[:head_end_2] + css_style + new_html[head_end_2:]

            # 🔁 5. Remove 'tex2jax_ignore' and 'mathjax_ignore' classes from any element
            # Target: <section class="tex2jax_ignore mathjax_ignore"> → remove both
            # Also target: div, span etc.
            
            # We'll use regex to match ANY element with these classes
            cleaned_html = re.sub(
                r'\bclass=["\'](?:tex2jax_ignore|mathjax_ignore)(?:[\s\w]*?["\'])',
                '',
                new_html,
                flags=re.IGNORECASE | re.DOTALL  # Case-insensitive, match any where
            )

            # 🔁 Optional: Also remove from <div> or <span>
            cleaned_html = re.sub(
                r'\bclass=["\'](?:tex2jax_ignore|mathjax_ignore)(?:[\s\w]*?["\'])',
                '',
                cleaned_html,
                flags=re.IGNORECASE | re.DOTALL
            )

            # ✅ Final: Save the clean HTML file
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(cleaned_html)

            print(f"✅ Fixed and saved: {file_path}")

if __name__ == "__main__":
    # 🔧 Change this to your actual directory
    input_dir = "./_build/html"
    output_dir = "./_build/html-patched"
    
    fix_jupyterbook_html(input_dir, output_dir)
