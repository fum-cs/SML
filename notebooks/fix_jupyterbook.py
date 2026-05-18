import os
import re
import shutil

def get_relative_path_to_root(file_path, output_dir):
    """
    Calculate the relative path from an HTML file to the output root directory
    """
    # Get the directory of the HTML file relative to output_dir
    rel_dir = os.path.relpath(os.path.dirname(file_path), output_dir)
    
    if rel_dir == '.':
        return ''
    else:
        # Count number of directories to go up
        depth = rel_dir.count(os.sep) + 1
        return '../' * depth

def fix_jupyterbook_html(input_dir, output_dir):
    """
    Fix all HTML files in input_dir to:
    1. Remove window.MathJax script (if present)
    2. Remove CDN MathJax script (if present)
    3. Add local MathJax script with correct relative path
    4. Add CSS style for body font
    5. Remove 'tex2jax_ignore' and 'mathjax_ignore' classes from any section/div
    6. Copy all non-HTML files and folders (except '_sources') to output_dir
    """
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # First, copy all contents except '_sources' and HTML files (will handle HTML separately)
    for root, dirs, files in os.walk(input_dir):
        # Skip '_sources' directory
        if '_sources' in dirs:
            dirs.remove('_sources')
        
        # Calculate relative path
        rel_path = os.path.relpath(root, input_dir)
        if rel_path == '.':
            target_dir = output_dir
        else:
            target_dir = os.path.join(output_dir, rel_path)
        
        # Create directory structure
        os.makedirs(target_dir, exist_ok=True)
        
        for file in files:
            input_file = os.path.join(root, file)
            
            # Handle HTML files separately (they need fixing)
            if file.endswith('.html'):
                output_file = os.path.join(target_dir, file)
                
                with open(input_file, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                
                # 🔁 1. Remove window.MathJax script (if exists)
                html_content = re.sub(
                    r'<script\s+[^>]*window\.MathJax\s*=\s*{[^}]+}</script>',
                    '', 
                    html_content, 
                    flags=re.IGNORECASE | re.DOTALL
                )
                
                # 🔁 2. Remove CDN MathJax script (if exists)
                html_content = re.sub(
                    r'<script[^>]*src="https://cdn\.jsdelivr\.net/npm/mathjax@3/es5/tex-mml-chtml\.js"[^>]*></script>',
                    '', 
                    html_content, 
                    flags=re.IGNORECASE | re.DOTALL
                )
                
                # 🔁 3. Remove other MathJax CDN variations
                html_content = re.sub(
                    r'<script[^>]*src="https://cdn\.jsdelivr\.net/npm/mathjax@[0-9]+/es5/[^"]*"[^>]*></script>',
                    '', 
                    html_content, 
                    flags=re.IGNORECASE | re.DOTALL
                )
                
                # 🔁 4. Calculate correct relative path to mathjax folder
                rel_path_to_root = get_relative_path_to_root(output_file, output_dir)
                mathjax_src = f'{rel_path_to_root}mathjax/MathJax.js?config=TeX-AMS-MML_SVG'
                
                # Add local MathJax script (inside <head>)
                mathjax_script = f'''
<script type="text/javascript" src="{mathjax_src}"></script>
'''
                
                # Find position to insert before </head> or at top of head
                head_end = html_content.find('</head>')
                if head_end == -1:
                    print(f"⚠️ Missing </head> in {file} — cannot inject MathJax")
                    # Still copy the file as is
                    shutil.copy2(input_file, output_file)
                    continue
                
                # Insert script BEFORE closing </head>
                new_html = html_content[:head_end] + mathjax_script + html_content[head_end:]
                
                # 🔁 5. Add CSS style (body font)
                css_style = '''
<style>
body { font-family: Arial, sans-serif, Asana-Math; }
</style>
'''
                
                # Find end of <head> to insert CSS
                head_end_2 = new_html.find('</head>')
                if head_end_2 == -1:
                    print(f"⚠️ Missing </head> in {file} — cannot inject style")
                    shutil.copy2(input_file, output_file)
                    continue
                
                new_html = new_html[:head_end_2] + css_style + new_html[head_end_2:]
                
                # 🔁 6. Remove 'tex2jax_ignore' and 'mathjax_ignore' classes from any element
                # More robust pattern to match class attributes containing these classes
                cleaned_html = re.sub(
                    r'class="([^"]*?)"?\b(?:tex2jax_ignore|mathjax_ignore)\b\s*"?([^"]*?)"?',
                    lambda m: f'class="{m.group(1)}{m.group(2)}"'.strip().replace('  ', ' ').rstrip('"').replace('class=""', '').replace('class=" "', ''),
                    new_html,
                    flags=re.IGNORECASE
                )
                
                # Remove empty class attributes
                cleaned_html = re.sub(r'class=""', '', cleaned_html)
                cleaned_html = re.sub(r'class="\s+"', '', cleaned_html)
                
                # Also handle single quotes
                cleaned_html = re.sub(
                    r"class='([^']*?)'?\b(?:tex2jax_ignore|mathjax_ignore)\b\s*'?([^']*?)'?",
                    lambda m: f"class='{m.group(1)}{m.group(2)}'".strip().replace('  ', ' ').rstrip("'").replace("class=''", ""),
                    cleaned_html,
                    flags=re.IGNORECASE
                )
                
                # ✅ Final: Save the clean HTML file
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(cleaned_html)
                
                # Print with relative path info
                if rel_path == '.':
                    print(f"✅ Fixed and saved: {file} (MathJax path: {mathjax_src})")
                else:
                    print(f"✅ Fixed and saved: {rel_path}/{file} (MathJax path: {mathjax_src})")
            
            else:
                # Copy non-HTML files as-is
                output_file = os.path.join(target_dir, file)
                shutil.copy2(input_file, output_file)
                if rel_path == '.':
                    print(f"📄 Copied: {file}")
                else:
                    print(f"📄 Copied: {rel_path}/{file}")

def copy_assets_and_resources(input_dir, output_dir):
    """
    Copy additional resources like mathjax folder, static assets, etc.
    """
    # Copy mathjax folder if it exists in the source
    mathjax_src = os.path.join(input_dir, 'mathjax')
    mathjax_dst = os.path.join(output_dir, 'mathjax')
    
    if os.path.exists(mathjax_src) and os.path.isdir(mathjax_src):
        if os.path.exists(mathjax_dst):
            shutil.rmtree(mathjax_dst)
        shutil.copytree(mathjax_src, mathjax_dst)
        print(f"✅ Copied mathjax folder to {mathjax_dst}")
    else:
        print(f"⚠️ mathjax folder not found at {mathjax_src}. Make sure local MathJax is available.")
    
    # Copy _static folder (usually contains CSS, JS, images)
    static_src = os.path.join(input_dir, '_static')
    static_dst = os.path.join(output_dir, '_static')
    
    if os.path.exists(static_src) and os.path.isdir(static_src):
        if os.path.exists(static_dst):
            shutil.rmtree(static_dst)
        shutil.copytree(static_src, static_dst)
        print(f"✅ Copied _static folder to {static_dst}")
    
    # Copy _images folder if it exists
    images_src = os.path.join(input_dir, '_images')
    images_dst = os.path.join(output_dir, '_images')
    
    if os.path.exists(images_src) and os.path.isdir(images_src):
        if os.path.exists(images_dst):
            shutil.rmtree(images_dst)
        shutil.copytree(images_src, images_dst)
        print(f"✅ Copied _images folder to {images_dst}")

if __name__ == "__main__":
    # 🔧 Configuration
    input_dir = "./_build/html"
    output_dir = "./_build/html-patched"
    
    print("🚀 Starting HTML processing...")
    print(f"📁 Input directory: {input_dir}")
    print(f"📁 Output directory: {output_dir}")
    print("-" * 50)
    
    # Process all HTML files and copy other files
    fix_jupyterbook_html(input_dir, output_dir)
    
    print("-" * 50)
    print("📦 Copying additional assets...")
    
    # Copy essential assets
    copy_assets_and_resources(input_dir, output_dir)
    
    print("-" * 50)
    print("✨ Done! All files processed and saved to:", output_dir)
    print("\n💡 Note: Make sure the 'mathjax' folder exists in the output root directory.")
    print("   The script automatically calculates correct relative paths for all HTML files.")