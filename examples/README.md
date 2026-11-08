# نقشهٔ پروفایل‌های نمونه

[شروع](../README.md) · [اجرا و خروجی](../docs/usage.md) · [فرض‌های پژوهشی](../docs/README.md)

هر فایل JSON ورودی یک اجرای مستقل است. نسخهٔ اصلی را حفظ و برای مطالعه کپی کنید. همهٔ فایل‌ها برای sensing عمومی یا اجرای کوتاه مناسب نیستند؛ schema، execution و study را پیش از اجرا بخوانید.

| پوشه | هدف و نقطهٔ ورود |
| --- | --- |
| [lab006](lab006/) | شروع: prescribed_reference.json فقط diagnostics؛ experimental_near_resonant.json اجرای محدود IF/AoA؛ experimental_reference.json برای تشخیص کانال |
| [lab007](lab007/) | schema 2، environment و spectrum، prescribed/experimental sensing و اسکن یکنواخت/غیریکنواخت |
| [atom001](atom001/)، [atom002](atom002/) | پروفایل‌های Rb85، Rb87 و Cs133؛ گونه را جایگزین خودکار فرض نکنید |
| [dop_td001](dop_td001/)، [dop_profile001](dop_profile001/)، [dop_conv001](dop_conv001/) | Doppler زمانی، quadrature تطبیقی و بررسی شبکهٔ سرعت |
| [num_ode002](num_ode002/) | تنظیمات صریح ODE برای سه گونه و حالت bounded warm |
| [relax002](relax002/) | radiative مدل selected_internal_spontaneous_e1 |
| [noise002](noise002/)، [noise003](noise003/) | Gaussian readout و campaign تکرار نویزی |
| [noise004](noise004/)، [noise005](noise005/) | PIN readout و RF drive noise |
| [noise006](noise006/)، [noise007](noise007/) | laser frequency noise و RIN |
| [noise008](noise008/)، [noise009](noise009/) | sweep نویز و آمار مطالعه |
| [noise010](noise010/)، [noise011](noise011/)، [noise012](noise012/) | همگرایی شبکه، مطالعهٔ IF و پاسخ میدان |
| [noise013](noise013/)، [noise014](noise014/) | IF-only با signal/null و مطالعهٔ آشکارسازی |
| [noise015](noise015/)، [noise016](noise016/)، [noise018](noise018/) | sampled sensitivity، بررسی عددی آشکارسازی و نقاط کار/ارزیابی candidate |
| [literature001](literature001/) | پروفایل‌های مرتبط با ادبیات و manifest.json برای دامنهٔ مقایسه |

manifest پوشهٔ literature001 ورودی ثابت آزمون است؛ آن را با خروجی تازه جایگزین نکنید. وجود پروفایل مرتبط با مقاله ادعای بازتولید دقیق آزمایش نیست. قابلیت‌ها و محدودیت‌ها در parser و اجراکنندهٔ مربوط تعیین می‌شوند؛ نام پوشه به‌تنهایی مرجع علمی نیست.
