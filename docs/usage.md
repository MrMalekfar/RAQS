# اجرای پروفایل و تفسیر خروجی

## انتخاب و اجرای معتبر

main.py و cli.py به cli.main می‌رسند. --profile الزامی است؛ بدون آن fallback قدیمی وجود ندارد. با وجود نمایش گزینه‌های قدیمی --n-elements، --target-aoa، --rf-frequency، --lo-rf-frequency، --spacing و --cell-length در help، ترکیب آن‌ها با پروفایل رد می‌شود. ورودی را در یک کپی از JSON ویرایش کنید.

تابع load_run_profile در [run_profile.py](../rydberg_aoa/config/run_profile.py) schema را سخت‌گیرانه می‌خواند؛ کلید ناشناخته، تکراری، مقدار غیرمتناهی یا declaration ناقص پذیرفته نمی‌شود. نسخهٔ 1 برای prescribed فقط diagnostics و برای experimental اجرای sensing دارد. نسخهٔ 2 environment، spectrum و execution صریح را اضافه می‌کند و prescribed sensing را نیز می‌پذیرد. spectrum می‌تواند null باشد؛ محیط حرارتی به‌تنهایی Doppler زمانی را فعال نمی‌کند.

در prescribed نردبان را شما تعیین می‌کنید؛ در experimental کانال با preparation و bounds از RF انتخاب می‌شود. طول‌موج پیشنهادی در proposed_wavelengths_nm_not_applied خودکار اعمال نمی‌شود. reference در نام فایل به معنی نزدیک‌بودن همهٔ میدان‌ها به رزونانس نیست؛ prescribed_reference نمونهٔ تشخیصی و دارای detuning اپتیکی بزرگ است.

دستورهای شروع در [README اصلی](../README.md) آمده‌اند. --diagnostics-only workers و DoA معمول را اجرا نمی‌کند؛ برای study مجاز نیست. --no-plots تولید/نمایش شکل CLI را خاموش می‌کند. بدون --output-dir گزارش در ترمینال چاپ می‌شود؛ ذخیرهٔ فایل‌ها را صریح بخواهید. stdout را همیشه JSON خالص فرض نکنید؛ پیام‌های کتابخانه‌ها ممکن است همراه آن باشند.

## خواندن نتیجه

برای اجرای تک‌پروفایل، یک پوشهٔ خالی یا تازه به `--output-dir` بدهید. `profile.json` همان بایت‌هایی است که برنامه خوانده است. `manifest.json` آخرین فایل است و SHA-256 ورودی و خروجی‌ها، نسخهٔ وابستگی‌ها و وضعیت Git را ثبت می‌کند. اگر manifest وجود ندارد، ذخیرهٔ اجرا کامل نشده است؛ برای اجرای دوباره پوشهٔ دیگری انتخاب کنید. اگر Git در دسترس نباشد، فیلدهای مربوط به آن `null` هستند.

| بخش گزارش | تفسیر |
| --- | --- |
| atom، levels، applied_fields | گونه، تراز حل‌شده و میدان اعمال‌شده؛ با پروفایل تطبیق دهید |
| detunings_rad_per_us، rabi_rad_per_us | detuningهای probe/coupling/RF و Rabiهای probe/coupling/LO/signal |
| rf_orientation، signed_if_mhz | جهت انرژی RF و اختلاف علامت‌دار signal−LO؛ یکی نیستند |
| decay_rate_basis، radiative_model | مبنای نرخ و دامنهٔ مدل radiative |
| diagnostics_only، sensing | تشخیص در برابر اجرای زمانی؛ نبود sensing در diagnostics طبیعی است |
| sensing | زاویهٔ واقعی/تخمینی، تعداد مسیر و نمونه و تنظیمات اجرای مؤثر |
| if_measurement | سنجه‌های استخراج IF؛ واحد و تعریف هر سنجه را از همان بخش بخوانید |
| noise_status و پرچم‌های included | اثر اجراشده را از declaration جدا می‌کنند |
| temporal_ode | تنظیمات و وضعیت اجرای حل؛ برای ادعای همگرایی کافی نیست |

در نمونهٔ آغازین نویز، Zeeman و ensemble Doppler فعال نیستند. برابری زاویه در این نمونهٔ ایده‌آل و تک‌منبع، دقت آماری یا عملکرد در نویز را تعیین نمی‌کند. پارامتر target_azimuth_deg ورودی حقیقت شبیه‌سازی است، دادهٔ تجربی نیست.

تابع save_profile_results در [profile_execution.py](../rydberg_aoa/simulation/profile_execution.py) دادهٔ تک‌اجرا را در simulation_arrays.npz می‌نویسد. برای مسیر AoA معمول:

- tlist زمان µs و time_traces به ترتیب مسیر، سپس زمان است.
- complex_responses پاسخ مختلط هر مسیر و array_positions موقعیت‌های متناظر است.
- if_snapshots ماتریس حسگر × پنجره و sample_covariance کوواریانس همین داده است.
- fft_frequency_path_* و fft_complex_path_* دادهٔ FFT هر مسیرند؛ طیف FFT را با تخمین دقیق IF یکی نگیرید.
- angles و power فقط برای تخمین‌گر دارای طیف ذخیره می‌شوند؛ ESPRIT و Root-MUSIC الزاماً منحنی زاویه‌ای ندارند.
- spectrum_*، trial_* و آرایه‌های detector فقط در مسیر مرتبط حاضرند. واحد trace نویزی را از sensing_trace_units یا receipt آشکارساز بخوانید؛ همهٔ traceها جریان الکتریکی نیستند.

برای خواندن آرایه‌ها با NumPy از np.load روی مسیر فایل استفاده کنید و ابتدا نام‌ها و shapeها را بررسی کنید. خروجی IF-only قرارداد مستقل در [if_only.py](../rydberg_aoa/simulation/if_only.py) دارد؛ وجود آرایه‌های AoA را برای آن فرض نکنید.

## study و نگهداری داده

CLI برای پروفایل study از [execute_study](../rydberg_aoa/simulation/study_execution.py) استفاده می‌کند. [study_output.py](../rydberg_aoa/simulation/study_output.py) گزارش study_report.json، پروفایل study_profile.json و زیرپوشه‌های نقاط/trialهای متناسب با نوع مطالعه را می‌نویسد. برخی مسیرها calibration و null_validation جدا دارند. writer از مخلوط‌کردن manifest موجود با مطالعهٔ تازه جلوگیری می‌کند؛ بااین‌حال همیشه پوشهٔ تازه بدهید. نبود گزارش نهایی پس از شکست، موفقیت مطالعه نیست.

همراه نتیجه، JSON اصلی، نسخهٔ کد و محیط و تنظیمات عددی را نگه دارید؛ [راهنمای پژوهشی](README.md) فهرست بازتولیدپذیری را توضیح می‌دهد.

## عیب‌یابی

- خطای ساخت ARC همراه readonly database: دسترسی نوشتن پایگاه کاربر را بررسی کنید؛ مشکل مجوز را با تغییر اتم یا دادهٔ ساختگی حل نکنید.
- PhysicsConfigurationError: declaration، q، mJ، bounds و سازگاری گزینه‌ها را اصلاح کنید؛ خوانده‌شدن JSON اثبات حل‌پذیری فیزیک نیست.
- خطای solver/worker: زنجیرهٔ خطا و تنظیمات را حفظ کنید؛ tolerance یا baseline را صرفاً برای عبور آزمون تغییر ندهید.
- اجرای طولانی: تعداد مسیر، نمونه، کلاس سرعت، نقاط study و trialها هزینه را افزایش می‌دهند؛ ابتدا نمونهٔ آغازین را اجرا کنید.


## نسخهٔ پروفایل و قرارداد نویز

نسخهٔ اصلی پروفایل و نسخهٔ بلوک `noise` مستقل‌اند. نمونه‌های اجرایی ثبت‌شده از نسخهٔ اصلی ۲ استفاده می‌کنند. نسخهٔ ۱ برای ورودی‌های قدیمی همچنان خوانده می‌شود. در نسخهٔ ۲، `environment: null` یعنی محیطی تعریف نشده و نرخ‌های صریح اجرای زمانی حفظ می‌شوند. `spectrum: null` یعنی محاسبهٔ طیف درخواست نشده است. در پروفایل prescribed تشخیصی، `execution: null` اجرای زمانی را غیرفعال می‌کند. طیف به محیط و اجرای صریح نیاز دارد و بودجهٔ حرارتی بدون اجرا پذیرفته نمی‌شود. گزارش اجرا نسخهٔ ورودی را در `input_schema_version` ثبت می‌کند.

نمونه‌های نویزی ثبت‌شده از `noise.schema_version: 4` استفاده می‌کنند. نسخه‌های ۱ تا ۳ همچنان خوانده می‌شوند. در قالب جدید:

- `enabled` کل بلوک نویز را کنترل می‌کند. `seed` و `trials` بین بخش‌ها مشترک‌اند.
- `drive` یا `null` است یا `settings` و `sources` را دارد. هر منبع RF یا لیزر کلید `enabled` و پارامترهای خودش را حفظ می‌کند.
- `readout` یا `null` است یا `model: "gaussian"` یا `model: "pin"` و منابع همان مدل را دارد. PIN به بلوک `detector` نیاز دارد. هنگام ترکیب با drive، `readout.enabled` خوانش را مستقل کنترل می‌کند. برای خوانش مستقل از `noise.enabled` استفاده می‌شود.
- خاموش‌کردن همهٔ منابع PIN با خاموش‌کردن آشکارساز فرق دارد. PIN روشن با منابع خاموش همچنان مسیر تبدیل آشکارساز را اجرا می‌کند. Gaussian و PIN دو انتخاب خوانش‌اند و هم‌زمان در یک پروفایل اجرا نمی‌شوند.

نمونهٔ Gaussian در [noise002](../examples/noise002/prescribed_gaussian.json) و نمونهٔ drive همراه PIN در [noise005](../examples/noise005/prescribed_rf_pin.json) قرار دارد. نسخهٔ جدید JSON به همان مدل‌های اجرایی قبلی نگاشت می‌شود. قالب گزارش نویز و شناسه‌های جریان تصادفی حفظ شده‌اند.
