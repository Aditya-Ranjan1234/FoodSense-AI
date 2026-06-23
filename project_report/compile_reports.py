import os
import subprocess
import shutil

report_dir = r"d:\6th Sem\idp\project_report"
sty_path = os.path.join(report_dir, "ecproject.sty")
pdflatex_path = r"C:\Users\OMEN\AppData\Local\Programs\MiKTeX\miktex\bin\x64\pdflatex.exe"
bibtex_path = r"C:\Users\OMEN\AppData\Local\Programs\MiKTeX\miktex\bin\x64\bibtex.exe"

def run_cmd(args):
    print(f"Running: {' '.join(args)}")
    subprocess.run(args, cwd=report_dir, check=False)

# 1. Compile standard green report
print("--- COMPILING GREEN REPORT ---")
run_cmd([pdflatex_path, "-interaction=nonstopmode", "IDP_Report.tex"])
run_cmd([bibtex_path, "IDP_Report"])
run_cmd([pdflatex_path, "-interaction=nonstopmode", "IDP_Report.tex"])
run_cmd([pdflatex_path, "-interaction=nonstopmode", "IDP_Report.tex"])

# Keep a copy of standard green report
shutil.copy(os.path.join(report_dir, "IDP_Report.pdf"), os.path.join(report_dir, "Report_Green.pdf"))

# 2. Modify sty file for B&W version
print("--- SWITCHING TO B&W SETTINGS ---")
with open(sty_path, "r", encoding="utf-8") as f:
    sty_content = f.read()

# Replace green fill with white fill
bw_content = sty_content.replace(
    r"\fill[fill={rgb,255:red,170;green,211;blue,177}] (current page.north west) rectangle (current page.south east);",
    r"\fill[fill=white] (current page.north west) rectangle (current page.south east);"
)
# Replace cover logo with white colorbox and BW logo
bw_content = bw_content.replace(
    r"{\setlength{\fboxsep}{0pt}\colorbox{covergreen}{\includegraphics[width=1\textwidth]{Figures/RV_newLogo}}\par}",
    r"{\setlength{\fboxsep}{0pt}\colorbox{white}{\includegraphics[width=1\textwidth]{Figures/RV_newLogo_BW}}\par}"
)

# Write temp B&W sty file
with open(sty_path, "w", encoding="utf-8") as f:
    f.write(bw_content)

try:
    # 3. Compile B&W report
    print("--- COMPILING B&W REPORT ---")
    run_cmd([pdflatex_path, "-interaction=nonstopmode", "IDP_Report.tex"])
    run_cmd([pdflatex_path, "-interaction=nonstopmode", "IDP_Report.tex"])
    
    # Save as Report_BW.pdf
    shutil.copy(os.path.join(report_dir, "IDP_Report.pdf"), os.path.join(report_dir, "Report_BW.pdf"))
finally:
    # 4. Restore original sty file
    print("--- RESTORING ORIGINAL GREEN SETTINGS ---")
    with open(sty_path, "w", encoding="utf-8") as f:
        f.write(sty_content)

# 5. Recompile green report to restore IDP_Report.pdf
print("--- RESTORING GREEN IDP_REPORT.PDF ---")
run_cmd([pdflatex_path, "-interaction=nonstopmode", "IDP_Report.tex"])
run_cmd([pdflatex_path, "-interaction=nonstopmode", "IDP_Report.tex"])

# Clean up temporary Green report file (now standard IDP_Report.pdf is the Green one)
if os.path.exists(os.path.join(report_dir, "Report_Green.pdf")):
    shutil.copy(os.path.join(report_dir, "Report_Green.pdf"), os.path.join(report_dir, "IDP_Report.pdf"))
    os.remove(os.path.join(report_dir, "Report_Green.pdf"))

print("--- DONE ---")
