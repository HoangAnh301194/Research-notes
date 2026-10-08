import json
import sys

def main():
    notebook_path = r'D:\PTIT\NCKH\Research-notes\Research_Report\261007\train_failed_echoch29.ipynb'
    output_path = r'D:\PTIT\NCKH\Research-notes\Research_Report\261007\outputs.txt'
    
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    with open(output_path, 'w', encoding='utf-8') as fout:
        for cell in nb.get('cells', []):
            if cell.get('cell_type') == 'code':
                for out in cell.get('outputs', []):
                    if out.get('output_type') == 'stream':
                        fout.write("".join(out.get('text', [])))
                    elif out.get('output_type') == 'execute_result':
                        data = out.get('data', {})
                        if 'text/plain' in data:
                            fout.write("".join(data['text/plain']))
                            fout.write("\n")

if __name__ == '__main__':
    main()
