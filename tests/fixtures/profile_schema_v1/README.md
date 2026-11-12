# Original LAB-006 inputs

Source: commit `b757ca9`, under `examples/lab006/`. Units and settings are those declared in each original profile. SHA-256 of the original bytes:

- `experimental_reference.json`: `f201db181a9efb54b60aa5b8cf17f7d850cd48dddd33b209aa7fe75d6bcf0317`
- `experimental_near_resonant.json`: `51dd30930e34b59114a5ade36370562207322baa68c55a88d2439aa2a3a46e3b`
- `prescribed_reference.json`: `4188cbebe7c62d9b08a1e19f73db73bea18694c8470161b3343847a74dd5819b`

رگرسیون کانال آماده، SHA-256 ورودی آخر را کنترل و آن را با ARC تعریف‌شده
در `requirements.txt` (نسخهٔ 3.9.0 در زمان رفع یافته‌های حسابرسی) حل می‌کند.
چگالی مرجع از همیلتونی در چارچوب چرخان و معادلهٔ لیندبلاد با ردِ واحد
مستقل در NumPy محاسبه می‌شود. واحد فرکانس رابی و detuning برابر rad/µs
و واحد نرخ رویداد برابر µs⁻¹ است. این حل به عملگرهای QuTiP یا خروجی
تاریخی و تولیدشدهٔ پروژه تکیه ندارد؛ ورودی‌های ARC بین دو حل مشترک‌اند
و آزمون، تطابق با دادهٔ تجربی را نمی‌سنجد.
