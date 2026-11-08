# RAQS — Rydberg AoA Simulation

شبیه‌سازی پژوهشی حسگر اتمی ریدبرگ برای EIT، دینامیک Lindblad، آشکارسازی heterodyne و تخمین زاویهٔ ورود RF. این مخزن کد اصلی، آزمون‌های فعال و پروفایل‌های نمونه را نگه می‌دارد. آرشیو پایان‌نامه، کتابخانهٔ PDF، نتایج اجراهای قبلی و محیط محلی توسعه در این نسخه قرار ندارند.

## نصب

محیط اعتبارسنجی‌شده Python 3.11 است. از ریشهٔ مخزن، در PowerShell:

```powershell
py -3.11 -m venv .venv
& ./.venv/Scripts/python.exe -m pip install -r requirements-dev.txt
```

برای اجرا بدون ابزارهای آزمون، `requirements.txt` کافی است. نسخه‌های مستقیم به محیط بررسی‌شده مقید شده‌اند؛ این فایل، قفل تمام وابستگی‌های گذرا یا تضمین اجرا روی همهٔ سیستم‌عامل‌ها نیست. ARC در نخستین استفاده ممکن است داده‌های اتمی را در پوشهٔ کاربر آماده کند؛ آن پوشه باید قابل نوشتن باشد.

## اجرای نمونه

```powershell
& ./.venv/Scripts/python.exe main.py --help
& ./.venv/Scripts/python.exe main.py --profile examples/lab006/prescribed_reference.json --diagnostics-only --no-plots --output-dir outputs/diagnostics/run01
& ./.venv/Scripts/python.exe main.py --profile examples/lab006/experimental_near_resonant.json --no-plots --output-dir outputs/sensing/run01
```

پروفایل صریح الزامی است. نمونهٔ اول فقط وضعیت اتمی و تنظیمات را گزارش می‌کند؛ نمونهٔ دوم مسیر sensing را اجرا می‌کند. `main.py` و `cli.py` به یک مسیر عمومی متصل‌اند. پوشهٔ خروجی را برای هر اجرا تازه انتخاب کنید. اجرای برنامه به نتایج اجرای قبلی وابسته نیست.

## آزمون

```powershell
& ./.venv/Scripts/python.exe -m pytest --collect-only -q
& ./.venv/Scripts/python.exe -m pytest tests/test_lab006_live_entrypoint.py -q
& ./.venv/Scripts/python.exe -m pytest tests -m "not slow" -q
```

اجرای همهٔ آزمون‌ها می‌تواند پرهزینه باشد؛ `not slow` تضمین سریع‌بودن تمام موارد باقی‌مانده نیست. آرشیو پژوهشی و آزمون‌های وابسته به خروجی‌های قدیمی جزو این مجموعه نیستند. عبور آزمون، تأیید تجربی مدل یا اثبات پوشش همهٔ فرض‌های علمی نیست.

[راهنمای استفاده](docs/usage.md)، [معماری](architecture.md)، [نقشهٔ مثال‌ها](examples/README.md) و [قواعد توسعه](AGENTS.md) جزئیات بیشتر را توضیح می‌دهند. مجوز این مخزن [MIT](LICENSE) است؛ همان مجوزی که مالک مخزن پیش از انتشار این نسخه انتخاب کرده است.
