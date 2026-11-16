@echo off
chcp 65001 >nul
echo ==============================================================================
echo  🚀 MUKESH AI - QWEN2-1.5B UNCENSORED CODING MODEL PIPELINE
echo  100%% Local & Self-Contained (No External API Needed)
echo ==============================================================================
echo.

echo [चरण 1/4] बेस मॉडल को डीक्वांटाइज़ कर Hugging Face फॉर्मेट में तैयार करना...
python 01_dequantize_base.py
if %errorlevel% neq 0 (
    echo ❌ चरण 1 विफल रहा।
    pause
    exit /b %errorlevel%
)

echo.
echo [चरण 2/4] कोडिंग क्षमता संवर्धन + 'मुकेश AI' पहचान इंजेक्शन...
python 02_enhance_code_and_identity.py
if %errorlevel% neq 0 (
    echo ❌ चरण 2 विफल रहा।
    pause
    exit /b %errorlevel%
)

echo.
echo [चरण 3/4] मैथमेटिकल एब्लिटरेशन (100%% Uncensored बनाना)...
python 03_uncensor_abliterate.py
if %errorlevel% neq 0 (
    echo ❌ चरण 3 विफल रहा।
    pause
    exit /b %errorlevel%
)

echo.
echo [चरण 4/4] Hugging Face पैकेजिंग और मॉडल कार्ड (README.md) निर्माण...
python 04_package_huggingface.py
if %errorlevel% neq 0 (
    echo ❌ चरण 4 विफल रहा।
    pause
    exit /b %errorlevel%
)

echo.
echo ==============================================================================
echo  🎉 सभी चरण सफलतापूर्वक 100%% पूरे हुए!
echo  📁 आपका फाइनल मॉडल यहाँ है:
echo     c:\Users\KRUNAL\Desktop\qwen_mukesh_improve\Mukesh-AI-Coder-1.5B-Uncensored
echo.
echo  🚀 मॉडल से चैट करने के लिए चलाएँ:
echo     python chat_cli.py   (या डबल क्लिक करें chat.bat)
echo.
echo  🧪 ऑटोमेटेड टेस्ट चलाने के लिए:
echo     python test_model.py (या डबल क्लिक करें test.bat)
echo.
echo  🌐 Hugging Face पर अपलोड करने के लिए चलाएँ:
echo     python upload_to_hf.py --repo USERNAME/Mukesh-AI-Coder-1.5B-Uncensored --token hf_yourToken
echo ==============================================================================
pause
