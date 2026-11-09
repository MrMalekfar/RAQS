# راهنمای پژوهشی

[شروع](../README.md) · [اجرا و خروجی](usage.md) · [مثال‌ها](../examples/README.md) · [آزمون‌ها](../tests/README.md) · [معماری](../architecture.md)

The English [Research Workbench guide](gui.md), [architecture and roadmap](gui-plan.md), and [verification report](gui-verification.md) describe the implemented desktop workflow and its explicit limits.

## مدل و مرز تفسیر

تابع create_sensor_model در [Hamiltonian](../rydberg_aoa/physics/hamiltonian/create_sensor_model.py) مدل چهارترازی ground → excited → rydberg1 ↔ rydberg2 را در RWA و با ℏ=1 می‌سازد. حل Lindblad و تبدیل coherence پروب به transmission ورودی پردازش IF را فراهم می‌کند. مدل معمول یک کانال مغناطیسی آماده‌شده را فرض می‌کند؛ توزیع جمعیت همهٔ زیرترازها را از فرایند آماده‌سازی پیش‌بینی نمی‌کند.

قرارداد QN-001 در [laboratory.py](../rydberg_aoa/config/laboratory.py) و [prepared_channel.py](../rydberg_aoa/arc/prepared_channel.py)، حالت HFS با F/mF و FS با j/mJ را جدا نگه می‌دارد. mF را نمی‌توان جای mJ گذاشت. قطبش کروی q و محور کوانتش صریح‌اند؛ کانال با اسکن mJ برای بزرگ‌کردن dipole انتخاب نمی‌شود. fs_centroid قرارداد انرژی اپتیکی نمونهٔ آغازین است؛ برچسب HFS به‌تنهایی به معنی لحاظ تمام جابه‌جایی‌های HFS در انرژی نیست.

ARC انرژی، dipole و نرخ را تأمین می‌کند. [atom_species.py](../rydberg_aoa/config/atom_species.py) گونه‌های Rb85، Rb87 و Cs133 را می‌شناسد؛ نبود بلوک atom به Rb85 خالص می‌رسد. isotope_fraction مدل دینامیکی مخلوط نیست. مدل برخورد برای همهٔ گونه‌ها قابل تعمیم فرض نشده است. شکست ARC یا کانال نامعتبر باید خطا بماند.

مدل نرخ پیش‌فرض در decay_rate_basis مشخص است و branching مغناطیسی کامل نیست. مدل اختیاری selected_internal_spontaneous_e1 در [profile_radiative.py](../rydberg_aoa/config/profile_radiative.py) گراف داخلی E1 را جایگزین می‌کند؛ بازگشت از تمام فضای خارج مدل را حل نمی‌کند. تابع execute_profile در [profile_execution.py](../rydberg_aoa/simulation/profile_execution.py) sensing عمومی magnetic core را رد می‌کند؛ diagnostics می‌تواند زیرمسیر زمانی تخصصی درخواست‌شده را اجرا کند. بنابراین diagnostics همیشه کم‌هزینه یا صرفاً خواندن JSON نیست.

## واحدها و قراردادها

| کمیت | قرارداد |
| --- | --- |
| زمان | µs؛ duration_us و tlist |
| RF ورودی، طول‌موج لیزر | Hz و nm |
| offset لیزر و IF | MHz چرخه‌ای؛ IF = (signal − LO)/10⁶ با علامت |
| Rabi و detuning حل | rad/µs؛ تبدیل Hz برابر 2πf/10⁶ |
| نرخ رویداد، واپاشی و transit | µs⁻¹؛ تبدیل s⁻¹ فقط تقسیم بر 10⁶ |
| توان، میدان، فاصله | mW، V/m، و m برای موقعیت حسگر و طول سلول |
| beam_waist_mm | شعاع شدت Gaussian در 1/e² بر حسب mm؛ قطر FWHM نیست |

واحد هر تابع را مستقل بررسی کنید؛ linewidth زاویه‌ای با نرخ رویداد یکی نیست. transit در [collapse operators](../rydberg_aoa/physics/lindblad/create_transit_reset_collapse_ops.py) بازنشانی جمعیت/همدوسی است و نباید دوباره به pure dephasing افزوده شود. جهت انرژی RF، علامت detuning و ضریب 1/2 Hamiltonian را از نام ترازها حدس نزنید؛ rf_orientation جهت را ثبت می‌کند.

در [steering.py](../rydberg_aoa/doa/steering.py)، azimuth از +y به سمت +x است و جهت افقی (sin az, cos az, 0) است. فاز ورود k·r و steering برابر exp(-j k·r) است. snapshots شکل (تعداد حسگر، تعداد پنجره) دارند؛ ترتیب ردیف‌ها باید با positions_m یکسان باشد. [snapshots.py](../rydberg_aoa/doa/snapshots.py) پنجرهٔ انتهایی ناقص را کنار می‌گذارد؛ طول پنجره از قدرمطلق IF و projection از علامت آن استفاده می‌کند. برای ESPRIT و Root-MUSIC قیود ULA را بررسی کنید. پروفایل عمومی یک منبع RF با elevation صفر را مدل می‌کند؛ چند تخمین‌گر به معنی پشتیبانی از چند منبع در این pipeline نیست (config/doa_options.py).

## اثرها و بازتولیدپذیری

وجود دما یا بلوک noise اثبات اجرای همهٔ اثرها نیست. Doppler طیفی و زمانی جدا هستند؛ doppler_ensemble_included و temporal_doppler_ensemble_included را مستقل بخوانید. نویز readout، PIN، RF drive، laser frequency و RIN مسیرهای مختلف دارند؛ noise_status و receipt اجرای واقعی تعیین‌کننده‌اند. طیف مرجع الزاماً نویز زمانی همان trial را ندارد.

برای هر نتیجه پروفایل، commit و diff محلی، نسخهٔ Python و بسته‌ها، گونه/نسخهٔ ARC و منشأ داده‌های اتمی، seed و noise_random_plan، تنظیمات solver، شبکهٔ زمانی/سرعت و تعداد trial را نگه دارید. خروجی تک‌پروفایل همهٔ این اطلاعات را خودکار بایگانی نمی‌کند. seed ثابت جای بررسی همگرایی و تکرار آماری را نمی‌گیرد؛ تطابق کامل بین پلتفرم‌ها تضمین نشده است.

outputs/ حاصل اجرای تازه و خارج از Git است؛ آن را ورودی ثابت آزمون نکنید. fixture مرجع باید منشأ، نسخه، واحد و hash داشته باشد. دو baseline ریشه قرارداد رگرسیون‌اند و نباید برای رفع شکست بازنویسی شوند. manifest نمونه‌های ادبیات دامنهٔ مقایسه را تعیین می‌کند؛ نام مقاله اثبات بازتولید آزمایش نیست.

ارزیابی تجربی، optical pumping عمومی و گیرندهٔ کامل چندترازی کار مستقل آینده‌اند؛ این مستندات برای آن‌ها ادعای قابلیت موجود یا زمان‌بندی ندارد.
