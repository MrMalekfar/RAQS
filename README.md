# RAQS — Rydberg AoA Simulation

RAQS نرم‌افزاری پژوهشی برای مدل‌سازی پاسخ حسگر اتمی ریدبرگ به RF و تخمین زاویهٔ ورود (AoA) از چند مسیر حسگری است. مسیر اصلی، میدان‌های probe، coupling، LO و signal را به مدل اتمی، حل Lindblad، آشکارسازی heterodyne و پردازش DoA متصل می‌کند. کاربرد آن مطالعهٔ عددی EIT، پاسخ IF و اثر هندسه و نویز بر گیرندهٔ شبیه‌سازی‌شده است. عبور آزمون‌ها به معنی تأیید تجربی مدل یا تضمین دقت دستگاه واقعی نیست.

## شروع

از ریشهٔ مخزن، در PowerShell با Python 3.11:

```powershell
py -3.11 -m venv .venv
& ./.venv/Scripts/python.exe -m pip install -r requirements-dev.txt
& ./.venv/Scripts/python.exe -m pip check
& ./.venv/Scripts/python.exe main.py --help
& ./.venv/Scripts/python.exe main.py --profile examples/lab006/prescribed_reference.json --diagnostics-only --no-plots --output-dir outputs/diagnostics/run01
& ./.venv/Scripts/python.exe main.py --profile examples/lab006/experimental_near_resonant.json --no-plots --output-dir outputs/sensing/run01
```

برای مصرف بدون آزمون، requirements.txt کافی است. در Linux/macOS محیط را با python3.11 بسازید و از .venv/bin/python استفاده کنید؛ بررسی این بازنویسی روی Windows انجام شده است. نسخه‌های مستقیم pin شده‌اند، اما قفل کامل وابستگی‌های گذرا نیستند. ARC به پایگاه دادهٔ قابل نوشتن در پوشهٔ کاربر نیاز دارد.

اجرای اول فقط تشخیص تنظیمات و حل دادهٔ اتمی است؛ اجرای دوم sensing را اجرا می‌کند. در run_report.json نمونهٔ دوم، سه مسیر، ۶۰۱ نمونه در هر مسیر، IF برابر 0.1 MHz و زاویهٔ ورودی/تخمینی ‎-30° انتظار می‌رود. این تطابق آزمون مسیر اجراست و معیار دقت عمومی حسگر نیست. آرایه‌ها در simulation_arrays.npz ذخیره می‌شوند. برای هر اجرای تک‌پروفایل، پوشه‌ای تازه یا خالی انتخاب کنید. ورودی در profile.json ثبت می‌شود و وجود manifest.json نشانهٔ پایان موفق ذخیره است.

## دامنهٔ قابلیت‌ها

- پروفایل صریح prescribed برای نردبان اعلام‌شده و experimental برای انتخاب کانال در bounds سخت‌افزار؛ نام experimental به معنی دادهٔ اندازه‌گیری‌شده نیست.
- پاسخ چهارترازی prepared-channel، طیف EIT، استخراج IF و پردازش آرایه؛ Doppler، transit و نویز با فعال‌سازی و محدودیت‌های مستقل.
- phase beamforming، Bartlett، MVDR، MUSIC، ESPRIT و Root-MUSIC با قیود هندسه و snapshots خودشان.
- مطالعهٔ چندنقطه‌ای، تکرار نویزی، آشکارسازی و بررسی همگرایی؛ وجود این مسیرها تضمین اعتبار هر ترکیب تنظیمات نیست.

پروفایل عمومی یک منبع RF با elevation صفر را مدل می‌کند. مدل اصلی شبیه‌سازی کامل optical pumping، سلول چندترازی یا دینامیک مخلوط ایزوتوپی نیست. Zeeman در گزارش مسیر اصلی وارد نشده است. مسیر magnetic core تشخیصی است و گیرندهٔ عمومی چندترازی را جایگزین نمی‌کند.

[راهنمای پژوهشی](docs/README.md)، [اجرای پروفایل و تفسیر خروجی](docs/usage.md) و [نقشهٔ مثال‌ها](examples/README.md) جزئیات را توضیح می‌دهند.

## آزمون و توسعه

Launch the English desktop Research Workbench with `python -m raqs_gui`. See the [GUI guide](docs/gui.md) for optional installation, configuration, execution, result review and current limits. The workbench uses the existing physical model.

```powershell
& ./.venv/Scripts/python.exe -m pytest tests/test_lab006_live_entrypoint.py -q -rA -s
```

[README آزمون‌ها](tests/README.md) دستورهای گسترده‌تر و معنای skip را توضیح می‌دهد. برای توسعه از [معماری](architecture.md) و [AGENTS.md](AGENTS.md) شروع کنید.

این مخزن کد، آزمون و پروفایل‌های نمونه را نگه می‌دارد؛ آرشیو پایان‌نامه و نتایج تاریخی جزو آن نیستند. مجوز کد [MIT](LICENSE) است.
