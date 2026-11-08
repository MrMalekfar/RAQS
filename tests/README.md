# آزمون و توسعه

از ریشهٔ مخزن و با محیط نصب‌شده از requirements-dev.txt اجرا کنید:

```powershell
& ./.venv/Scripts/python.exe -m pytest --collect-only -q
& ./.venv/Scripts/python.exe -m pytest tests/test_lab006_live_entrypoint.py -q -rA -s
& ./.venv/Scripts/python.exe -m pytest tests -m "not slow" -q -ra
& ./.venv/Scripts/python.exe -m pytest tests -q -ra
```

دو دستور آخر دامنهٔ گسترده و کامل‌اند و ممکن است پرهزینه باشند. not slow فقط marker را حذف می‌کند و تضمین سرعت نیست. collection اجرای آزمون نیست؛ کد خروج صفر با skip نیز به معنی عبور همان مورد نیست.

## چه چیزی بررسی می‌شود؟

| نقطهٔ شروع | قرارداد |
| --- | --- |
| test_lab006_live_entrypoint.py | subprocess واقعی، ARC، diagnostics و sensing با spawn |
| test_lab006_run_profile.py، test_lab007_profile_integration.py | schema و ادغام پروفایل |
| test_arch001*، test_mode001* | مقداردهی صریح، rollback و مرز حالت‌ها |
| test_c001_rabi_contract.py، test_beam002_radius_contract.py | Rabi، dipole و تعریف شعاع |
| test_num_td001_density_validation.py، test_num_ode002_profile.py | اعتبار density و تنظیمات حل |
| test_doa_*، test_if_* | steering، snapshots، covariance و استخراج IF |
| test_noise*، test_dop*، test_dph* | قرارداد اثرها، مطالعه‌ها و transit |

fixture اتمی در [conftest.py](conftest.py) opt-in است؛ import جای initialize_atomic_physics را نمی‌گیرد (ARCH-001). atomic_physics مقداردهی واقعی می‌خواهد؛ published_atomic_state فقط برای آزمون مصرف‌کننده با ورودی مصنوعی است و صحت مقداردهی را ثابت نمی‌کند.

[atomic_init_support.py](atomic_init_support.py) فقط شکست مستند QN-001A در مرز HFS→FS بدون Rydberg mJ صریح را به skip با دلیل BLOCKED تبدیل می‌کند. سایر خطاها باید آشکار بمانند. skip را جدا از pass گزارش کنید.

این مجموعه به کد، پروفایل و baselineهای همین نسخه وابسته است؛ outputs/ ورودی ثابت آن نیست. دادهٔ مصنوعی صحت علمی مدل را ثابت نمی‌کند. آزمون زنده نیز به‌تنهایی اعتبار تجربی یا پوشش همهٔ شرایط کاری را اثبات نمی‌کند.

## ورود به توسعه

نقشهٔ مسیر عمومی در [architecture.md](../architecture.md) است: main.py → cli.main → load_run_profile → execute_profile یا execute_study. برای ورودی تازه ابتدا config/run_profile.py و آزمون parser؛ برای حل، physics/ و simulation/؛ برای پردازش آرایه، doa/ و simulation/doa_pipeline.py را بخوانید. وجود run_simulation_with_config قدیمی در cli.py به معنی مسیر عمومی توصیه‌شده نیست.

[AGENTS.md](../AGENTS.md) قراردادهای واحد، QN-001 و ARCH-001 را مشخص می‌کند. تغییر solver باید ordering، trace، Hermiticity، positivity و همگرایی مرتبط را بررسی کند. تغییر DoA باید ترتیب حسگر، فاز و شکل snapshots/covariance را حفظ کند. آزمون رفتار متأثر را اجرا و تعداد pass/fail/skip و خطای محیطی را جدا گزارش کنید؛ baseline، seed و tolerance ابزار سبزکردن آزمون نیستند.

## وضعیت بررسی این بازنویسی

بررسی در ۲۰۲۶-۱۱-۰۸ با Windows و Python 3.11.0 در محیط موجود انجام شد. pip check و pip install --dry-run -r requirements-dev.txt موفق بودند؛ نصب از صفر و سیستم‌عامل دیگر بررسی نشد. جمع‌آوری pytest تعداد ۲۷۴۲ مورد را یافت.

اجرای اولیهٔ آزمون ورودی در sandbox پیش از collection به علت SQLite فقط‌خواندنی ARC متوقف شد؛ این شکست محیطی pass یا skip نیست. اجرای مجدد با capture پیش‌فرض پس از سه موردِ عبورکرده متوقف ماند و قطع شد؛ نتیجهٔ کامل ندارد. اجرای تازه با دسترسی معمول ARC و گزینهٔ -s هر ۵ آزمون زنده را گذراند: ۵ pass، صفر fail و صفر skip. دستور پیشنهادی بالا همین اجرای بررسی‌شده است.

دستور زیر نیز اجرا شد و ۸۵ pass، صفر fail و صفر skip داشت:

```powershell
& ./.venv/Scripts/python.exe -m pytest tests/test_lab006_run_profile.py tests/test_lab007_profile_integration.py -q -ra
```

هر دو دستور نمونهٔ README و --help موفق بودند. گزارش sensing مقادیر مورد انتظار را داشت؛ شکل time_traces برابر (3, 601)، if_snapshots برابر (3, 2) و sample_covariance برابر (3, 3) بود. لینک‌های محلی و git diff --check نیز بررسی شدند. کل مجموعه و not slow اجرا نشدند؛ موفقیت این ۹۰ آزمون را به همهٔ ۲۷۴۲ مورد تعمیم ندهید. هشدارهای FutureWarning از QuTiP دیده شد؛ رفتار کتابخانه یا مدل برای رفع آن‌ها تغییر نکرد.
