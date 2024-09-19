import os
import re

# Function to convert LaTeX sections to Markdown headings
def convert_sections(content):
    # Convert \section{...} to # ...
    content = re.sub(r'\\section{(.+?)}', r'# \1', content)
    
    # Convert \subsection{...} to ## ...
    content = re.sub(r'\\subsection{(.+?)}', r'## \1', content)
    
    # Convert \subsubsection{...} to ### ...
    content = re.sub(r'\\subsubsection{(.+?)}', r'### \1', content)
    
    return content

# Function to convert abstract to Markdown
def convert_abstract(content):
    # Convert \begin{abstract} ... \end{abstract} to > ...
    content = re.sub(r'\\begin{abstract}', r'> ', content)
    content = re.sub(r'\\end{abstract}', r'', content)
    
    return content

# Function to convert LaTeX math to Markdown MathJax
def convert_equations(content):
    # Inline equations: $...$
    content = re.sub(r'\$(.+?)\$', r'$\1$', content)
    
    # Block equations: \[...\] or $$...$$
    content = re.sub(r'\\\[(.+?)\\\]', r'$$\1$$', content)
    content = re.sub(r'\$\$(.+?)\$\$', r'$$\1$$', content)
    
    return content

# Function to convert LaTeX image references to Markdown
def convert_images(content):
    # Convert \includegraphics[options]{path} to Markdown ![alt text](path)
    content = re.sub(r'\\includegraphics(\[.*?\])?{(.+?)}', r'![Image](\2)', content)
    return content

# Function to convert bibliography and citations
def convert_bibliography(content):
    # Regular expression pattern
    pattern = r'\\citep{([^}]+)}'
    
    def replacement(match):
        citekeys = [key.strip() for key in match.group(1).split(',')]
        if len(citekeys) > 1:
            pandoc_cites = '; '.join('@' + key for key in citekeys)
            return f'[[{pandoc_cites}]]'
        else:
            return f'[@{citekeys[0]}]'
    
    # Perform the replacement
    content = re.sub(pattern, replacement, content)
    return content

# replace comments with multiline comments
def replace_comments(content):
    # Convert %... to <!-- ... -->
    content = re.sub(r'%(.+)', r'<!-- \1 -->', content)
    
    return content

    
    

# convert italics
def convert_italics(content):
    # Convert \textit{...} to *...*
    content = re.sub(r'\\textit{(.+?)}', r'*\1*', content)

    return content
# convert bold
def convert_bold(content):
    # Convert \textbf{...} to **...**
    content = re.sub(r'\\textbf{(.+?)}', r'**\1**', content)

    return content

# Main function to process LaTeX files
def convert_latex_to_markdown(input_dir, output_dir):
    # Make sure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.tex'):
                # Read the LaTeX file
                with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Perform the conversions
                content = convert_equations(content)
                content = convert_images(content)
                content = convert_bibliography(content)
                content = convert_abstract(content)
                content = convert_sections(content)
                content = convert_italics(content)
                content = convert_bold(content)
                
                # Write the converted content to a Markdown file
                output_file = os.path.join(output_dir, file.replace('.tex', '.md'))
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(content)

# Run the conversion
if __name__ == "__main__":
    input_directory = '/Users/ppruthi/research/compositional_models/clear_paper'  # Replace with the path to your LaTeX project
    output_directory = '/Users/ppruthi/research/PhD_thesis'  # Replace with the path to your Obsidian vault
    os.makedirs(output_directory, exist_ok=True)
    
    convert_latex_to_markdown(input_directory, output_directory)
