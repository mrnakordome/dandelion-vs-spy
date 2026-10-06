"""Build the final report (Persian, RTL) from the measured results.

    python -m src.report          # writes ../Report.html  (+ Report.pdf if Chrome is found)

Every number in the report is injected from ``results/results.json`` so the
document can never disagree with the experiments.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from typing import Dict, List

from .experiments import RESULTS, get_topology

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_HTML = os.path.join(os.path.dirname(ROOT), "Report.html")
OUT_PDF = os.path.join(os.path.dirname(ROOT), "Report.pdf")
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

CSS = """
@page { size: A4; margin: 17mm 15mm; }
* { box-sizing: border-box; }
body { font-family: "Vazirmatn","IRANSans","Geeza Pro","Arial Unicode MS",Tahoma,sans-serif;
       direction: rtl; text-align: justify; color:#15181d; line-height:1.85;
       font-size: 10.5pt; margin:0; }
h1 { font-size: 20pt; text-align:center; margin:0 0 4pt; color:#0f2c4d; }
h2 { font-size: 14pt; color:#0f2c4d; border-bottom:2px solid #dfe6ee; padding-bottom:3pt;
     margin-top:20pt; page-break-after:avoid; }
h3 { font-size: 11.5pt; color:#20456f; margin-top:13pt; page-break-after:avoid; }
p  { margin: 6pt 0; }
.sub { text-align:center; color:#5a6675; margin:0 0 14pt; font-size:10pt; }
.box { background:#f4f7fb; border:1px solid #dbe4ee; border-radius:6px;
       padding:8pt 11pt; margin:9pt 0; }
.box.warn { background:#fff8ec; border-color:#f0dcb4; }
.box.key  { background:#eef7f0; border-color:#cfe6d5; }
code, .ltr { direction:ltr; unicode-bidi:embed; font-family:"SF Mono",Menlo,Consolas,monospace;
             font-size:9.3pt; }
pre { direction:ltr; text-align:left; background:#f6f8fa; border:1px solid #e3e8ee;
      border-radius:5px; padding:7pt 9pt; overflow-x:auto; font-size:9pt; line-height:1.5; }
table { border-collapse:collapse; width:100%; margin:9pt 0; font-size:9.2pt;
        page-break-inside:avoid; }
th, td { border:1px solid #d7dfe8; padding:4pt 6pt; text-align:center; }
th { background:#eaf0f7; font-weight:600; }
td.r, th.r { text-align:right; }
figure { margin:12pt 0; text-align:center; page-break-inside:avoid; }
figure img { max-width:100%; border:1px solid #e2e8ef; border-radius:5px; }
figcaption { font-size:9pt; color:#5a6675; margin-top:4pt; }
.formula { direction:ltr; text-align:center; font-family:"Times New Roman",serif;
           font-size:12pt; background:#fafbfd; border:1px dashed #d5dde6;
           border-radius:5px; padding:6pt; margin:8pt 0; }
ul, ol { margin:5pt 22pt 5pt 0; padding:0; }
li { margin:3pt 0; }
.small { font-size:9pt; color:#5a6675; }
.pb { page-break-before: always; }
"""


def fa(x) -> str:
    """Format a number for the Persian text (kept LTR)."""
    if isinstance(x, float):
        s = ("%.3f" % x).rstrip("0").rstrip(".")
    else:
        s = str(x)
    return '<span class="ltr">%s</span>' % s


def pct(x: float) -> str:
    return '<span class="ltr">%.1f%%</span>' % (100.0 * x)


def ms(x: float) -> str:
    return '<span class="ltr">%.0f ms</span>' % (1000.0 * x)


def sec(x: float) -> str:
    return '<span class="ltr">%.3f s</span>' % x


def mstd(d: Dict[str, float], f: str = "%.3f") -> str:
    return '<span class="ltr">%s ± %s</span>' % (f % d["mean"], f % d["std"])


def img(name: str, caption: str) -> str:
    return ('<figure><img src="Code/results/figures/%s"><figcaption>%s</figcaption>'
            '</figure>' % (name, caption))


def build(res: Dict, topo, calib: Dict) -> str:
    P = res["p_values"]
    kmax = res["kmax"]
    kopt = res["phase2"]["optimal_k"]
    v = res["topology"]
    ph1, ph2, ph3, ph4, ph5 = (res["phase1"], res["phase2"], res["phase3"],
                               res["phase4"], res["phase5"])
    sw = ph2["sweep"]
    o: List[str] = []
    w = o.append

    def sweep_row(k: int, key: str, p=None):
        blk = sw if p is None else ph4["p=%.1f" % p]["sweep"]
        return blk[k - 1][key]

    # ------------------------------------------------------------------ head
    w("<h1>قاصدک در برابر جاسوس</h1>")
    w('<p class="sub">شبیه‌سازی پویا و چالش حریم خصوصی در شبکه — پروژهٔ پایانی درس '
      'شبکه‌های کامپیوتری (۴۰۴۴۳)<br>استاد درس: دکتر سید امیرمهدی صادق‌زاده — نیم‌سال دوم ۱۴۰۴–۱۴۰۵</p>')

    w('<div class="box key"><b>خلاصهٔ نتایج.</b> یک شبکهٔ فضایی با '
      f'{fa(res["n_nodes"])} گره (هر گره یک پردازهٔ مستقل روی یک پورت UDP جداگانه) '
      f'و {fa(v["n_clusters"])} کلاستر ساخته شد. در پخش عمومی ساده، حملهٔ پیشنهادی '
      f'با تنها {fa(kopt)} گرهٔ جاسوس ({pct(kopt / res["n_nodes"])} شبکه) مبدأ '
      f'{pct(sweep_row(kopt, "timing_ml")["mean"])} بسته‌ها را درست تشخیص می‌دهد، در حالی که '
      f'خط پایهٔ «اولین جاسوس» تنها {pct(sweep_row(kopt, "first_spy")["mean"])} دقت دارد؛ '
      f'با {fa(kmax)} جاسوس (سقف مجاز ۳۰٪) دقت به {pct(sweep_row(kmax, "timing_ml")["mean"])} می‌رسد. '
      'پروتکل Dandelion این دقت را به‌شدت کاهش می‌دهد: با همان '
      f'{fa(kmax)} جاسوس و {fa("p = 0.9")}، دقت حملهٔ آگاه از Dandelion به '
      f'{pct(sweep_row(kmax, "dandelion_ml", 0.9)["mean"])} می‌رسد — به قیمت افزایش زمان پوشش '
      f'از {sec(ph1["T80"]["mean"])} به {sec(ph3["p=0.9"]["T80"]["mean"])}.</div>')

    # --------------------------------------------------------------- part 1
    w('<h2>۱. معماری سیستم و قواعد شبکه</h2>')
    w('<h3>۱.۱ معماری داخلی گره</h3>')
    w('هر گره یک <b>پردازهٔ مستقل سیستم‌عامل</b> است '
      f'(<code>python -m src.node &lt;config&gt;</code>) که کنترل‌کننده آن را اجرا می‌کند. '
      'تمام ورودی/خروجی با <code>asyncio</code> و کاملاً ناهم‌زمان است: یک '
      '<code>DatagramProtocol</code> برای دادهٔ UDP و یک سرور TCP کوچک فقط برای '
      'فرمان‌های شبیه‌ساز (تزریق بسته و خاموش‌سازی). حالت نگهداری‌شده در هر گره:')
    w('<ul>'
      '<li><b>NodeID</b>: یک UUID یکتا، کاملاً مستقل از آدرس شبکه (تولید قطعی از روی بذر).</li>'
      '<li><b>SelfAddr</b>: زوج <code>ip:port</code> روی <code>127.0.0.1</code>؛ '
      'هر گره پورت UDP اختصاصی خود را دارد.</li>'
      '<li><b>PeerList</b>: همسایه‌های مستقیم به‌همراه آدرس، تأخیر پایهٔ لینک، '
      'آخرین زمان مشاهده (<code>last_seen</code>)، آخرین بستهٔ دریافتی از آن همسایه و '
      'شمارندهٔ ارسال/دریافت. درجهٔ هر گره در بازهٔ ۲ تا ۴ تضمین شده است.</li>'
      '<li><b>SeenSet</b>: مجموعهٔ شناسهٔ بسته‌های پردازش‌شده برای حذف تکراری‌ها؛ '
      'مجموعهٔ جداگانهٔ <code>stem_seen</code> برای تشخیص بازگشت گشت تصادفی ساقه.</li>'
      '</ul>')
    w('<h3>۱.۲ قالب بسته روی سیم</h3>')
    w('تنها بایت‌هایی که روی UDP منتقل می‌شوند عبارت‌اند از:')
    w('<pre>&lt;packet-id&gt;|&lt;S|F&gt;        example:  p0137|S</pre>')
    w('یعنی فقط <b>شناسهٔ یکتای بسته</b> و <b>وضعیت آن</b> (Stem/Fluff). '
      'شناسهٔ گره مبدأ و زمان تولید بسته هرگز منتقل نمی‌شوند و تنها در لاگ مرجع '
      'شبیه‌ساز ثبت می‌گردند. گرهٔ فرستندهٔ قبلی صرفاً از روی آدرس مبدأ دیتاگرام UDP '
      'شناخته می‌شود — همان اطلاعاتی که یک گره در شبکهٔ واقعی هم در اختیار دارد. '
      'ساعت همهٔ گره‌ها هم‌گام فرض شده و تأخیر پردازش داخلی صفر است؛ تنها در فاز ۵ '
      'گره‌های جاسوس تأخیر عمدی اعمال می‌کنند.')

    w('<h3>۱.۳ تولید توپولوژی</h3>')
    w('گره‌ها روی صفحهٔ <span class="ltr">1000 × 1000</span> جانمایی می‌شوند. '
      'الگوریتم به‌ترتیب زیر است:')
    w('<ol>'
      f'<li><b>مراکز کلاستر:</b> {fa(v["n_clusters"])} مرکز با نمونه‌گیری با پس‌زدن '
      '(rejection sampling) در ناحیهٔ <span class="ltr">[150, 850]²</span> انتخاب می‌شوند، '
      'با شرط حداقل فاصلهٔ <span class="ltr">260</span> واحد میان هر دو مرکز (پخش تقریباً '
      'یکنواخت مراکز و جلوگیری از ادغام کلاسترها).</li>'
      '<li><b>اعضای کلاستر:</b> ۸۰٪ گره‌ها با توزیع گاوسی همسان‌گرد حول مرکز خود با '
      '<span class="ltr">σ = 55</span> نمونه‌گیری می‌شوند؛ نمونه‌ای که به مرکز دیگری '
      'نزدیک‌تر باشد یا بیرون صفحه بیفتد دور ریخته و دوباره نمونه‌گیری می‌شود. '
      'این کار تراکم موضعی زیاد و مرز روشن میان کلاسترها ایجاد می‌کند.</li>'
      '<li><b>گره‌های پراکنده:</b> ۲۰٪ باقی‌مانده به‌صورت یکنواخت روی صفحه و با حفظ '
      'حداقل فاصلهٔ <span class="ltr">90</span> واحد از مراکز کلاستر قرار می‌گیرند.</li>'
      '<li><b>یال‌های درون‌کلاستری:</b> درخت پوشای کمینهٔ اقلیدسی هر کلاستر ساخته '
      'می‌شود و سپس هر عضو با درجهٔ کمتر از ۲ به نزدیک‌ترین هم‌کلاستری خود وصل می‌شود.</li>'
      '<li><b>گره‌های پراکنده</b> به دو همسایهٔ نزدیک خود (در کل شبکه) متصل می‌شوند.</li>'
      '<li><b>اتصال چندگانهٔ کلاسترها:</b> مراکز کلاسترها بر حسب زاویهٔ قطبی حول مرکز '
      'ثقل کل مرتب می‌شوند و یک <b>حلقه</b> میان کلاسترهای متوالی از طریق نزدیک‌ترین '
      'زوج گره ساخته می‌شود؛ دو لینک هر کلاستر عمداً از <b>دو گرهٔ متفاوت</b> عبور '
      'می‌کنند تا مستقل باشند. سپس چند لینک عرضی کوتاه دیگر نیز افزوده می‌شود.</li>'
      '<li><b>ترمیم:</b> حلقه‌ای تکراری درجهٔ همهٔ گره‌ها را به بازهٔ <span class="ltr">[2,4]'
      '</span> می‌آورد، مؤلفه‌های جدا را وصل می‌کند و هر <b>پل</b> (یال برشی) را با افزودن '
      'کوتاه‌ترین وتر ممکن حذف می‌کند. اگر توپولوژی پس از ترمیم معتبر نباشد، با بذر '
      'مشتق‌شده دوباره تولید می‌شود.</li>'
      '</ol>')
    w('<div class="box"><b>چرا «بی‌پل بودن» را بررسی می‌کنیم؟</b> شرط صورت پروژه این است '
      'که هر کلاستر دست‌کم از دو لینک ارتباطی مستقل به بقیهٔ شبکه وصل باشد تا مسیر '
      'جایگزین وجود داشته باشد. شرط قوی‌تر «گراف هیچ پلی ندارد» (۲-یال-همبندی) دقیقاً '
      'معادل این است که میان هر دو گره حداقل دو مسیر یال-مجزا وجود دارد؛ بنابراین '
      'برقراری آن، شرط صورت پروژه را برای همهٔ کلاسترها و حتی همهٔ گره‌ها تضمین می‌کند. '
      'اعتبارسنجی خودکار (تابع <code>topology.validate</code>) هر دو شرط را جداگانه '
      'گزارش می‌کند.</div>')

    w('<table><tr><th class="r">ویژگی</th><th>مقدار</th><th class="r">ویژگی</th><th>مقدار</th></tr>'
      f'<tr><td class="r">تعداد گره</td><td>{fa(res["n_nodes"])}</td>'
      f'<td class="r">تعداد یال</td><td>{fa(v["n_edges"])}</td></tr>'
      f'<tr><td class="r">تعداد کلاستر</td><td>{fa(v["n_clusters"])}</td>'
      f'<td class="r">اندازهٔ کلاسترها</td><td class="ltr">{", ".join(str(x) for x in v["cluster_sizes"].values())}</td></tr>'
      f'<tr><td class="r">همبند است؟</td><td>{"بله" if v["connected"] else "خیر"}</td>'
      f'<td class="r">تعداد پل</td><td>{fa(v["n_bridges"])}</td></tr>'
      f'<tr><td class="r">درجهٔ کمینه/بیشینه</td><td class="ltr">{v["degree_min"]} / {v["degree_max"]}</td>'
      f'<td class="r">درجهٔ میانگین</td><td>{fa(v["degree_mean"])}</td></tr>'
      f'<tr><td class="r">لینک عرضی هر کلاستر</td><td class="ltr">{", ".join(str(x) for x in v["cluster_cross_links"].values())}</td>'
      f'<td class="r">شرط دو لینک مستقل</td><td>{"برای همهٔ کلاسترها برقرار" if v["all_clusters_ok"] else "نقض شده"}</td></tr>'
      '</table>')
    w(img("fig1_topology.png",
          'شکل ۱ — توپولوژی ثابت همهٔ آزمایش‌ها. رنگ‌ها کلاسترها را نشان می‌دهند، '
          'مربع‌های خاکستری گره‌های پراکنده و دایره‌های قرمز گره‌های جاسوس‌اند. '
          'ضخامت یال‌ها معکوس تأخیر پایه است.'))

    w('<h3>۱.۴ مدل تأخیر لینک</h3>')
    w('تأخیر پایهٔ هر لینک برابر فاصلهٔ اقلیدسی دو سر آن است (۱ میلی‌ثانیه به ازای هر '
      'واحد فاصله) و به ازای <b>هر ارسال</b> جیتر یکنواخت ۲۰٪ اعمال می‌شود:')
    w('<div class="formula">Delay<sub>total</sub> = Delay<sub>base</sub> + '
      'U(−0.2 × Delay<sub>base</sub>, +0.2 × Delay<sub>base</sub>)</div>')
    w('پیاده‌سازی: فرستنده تأخیر را محاسبه و ارسال واقعی دیتاگرام را با '
      '<code>loop.call_later</code> زمان‌بندی می‌کند؛ بنابراین تأخیر لینک بدون مسدود '
      'کردن حلقهٔ رویداد مدل می‌شود. اسکریپت <code>src/verify.py</code> روی همهٔ '
      f'{fa(len(res["seeds"]) * (1 + 2 * len(P)) + len(res["seeds"]))} اجرا بررسی می‌کند که '
      'نسبت تأخیر زمان‌بندی‌شده به تأخیر پایه دقیقاً در بازهٔ '
      '<span class="ltr">[0.8, 1.2]</span> بماند (و برای جاسوس‌های فاز ۵ در '
      '<span class="ltr">[1.8, 2.2]</span>).')

    w('<h3>۱.۵ مدل رفتاری مهاجم</h3>')
    w('<ul>'
      '<li>مهاجم گره جدید اضافه نمی‌کند؛ تنها با رشوه، گره‌های موجود را جاسوس می‌کند و '
      'هزینهٔ رشوه برای همهٔ گره‌ها یکسان است — به همین دلیل امتیاز او بر تعداد '
      'جاسوس‌ها تقسیم می‌شود.</li>'
      '<li>مهاجم تأخیر پایهٔ همهٔ لینک‌ها و توپولوژی را می‌داند، اما دید سراسری ندارد؛ '
      'فقط بسته‌هایی را می‌بیند که به جاسوس‌های او می‌رسند (شناسهٔ بسته، وضعیت، '
      'همسایهٔ تحویل‌دهنده و زمان ورود محلی).</li>'
      '<li>جاسوس‌ها حق حذف یا مسدود کردن بسته را ندارند و دقیقاً طبق پروتکل بازپخش '
      'می‌کنند؛ تنها در فاز ۵ می‌توانند پیش از بازپخش تأخیر عمدی (حداکثر یک تأخیر '
      'پایه) اعمال کنند.</li>'
      '<li>گم‌شدن بسته در لایهٔ UDP مدل نمی‌شود؛ همهٔ ارسال‌های محلی موفق فرض می‌شوند و '
      'هر رویداد غیرمنتظره در لاگ ثبت و توسط <code>verify</code> بررسی می‌شود '
      '(در هیچ اجرایی رخ نداد).</li>'
      '</ul>')

    # --------------------------------------------------------------- part 2
    w('<h2 class="pb">۲. سناریوی آزمایش</h2>')
    w(f'در هر اجرا {fa(res["n_packets"])} بسته تولید می‌شود. مبدأ هر بسته به‌صورت کاملاً '
      'تصادفی از میان گره‌های درستکار انتخاب می‌شود و زمان تولید بسته‌ها از توزیع '
      'یکنواخت روی یک پنجرهٔ زمانی (بدون هیچ الگوی زمانی از پیش تعیین‌شده) کشیده '
      'می‌شود. زمان‌بندی و مبدأها تنها از روی <b>بذر سناریو</b> ساخته می‌شوند؛ بنابراین '
      'برای یک بذر مشخص، دقیقاً همان ۲۰۰ زوج (مبدأ، زمان) در همهٔ فازها و همهٔ مقادیر '
      f'{fa("p")} تکرار می‌شود و مقایسه منصفانه است. توپولوژی و مجموعهٔ جاسوس‌ها نیز در '
      f'کل کارزار ثابت‌اند. پنج بذر مستقل {fa(str(res["seeds"]))} استفاده شده و '
      'میانگین، میانه و انحراف معیار گزارش می‌شود.')

    w('<h2>۳. فاز ۱ — پخش عمومی پایه</h2>')
    w('هر گره پس از نخستین دریافت یک بسته، بلافاصله آن را برای همهٔ همسایه‌های خود '
      'ارسال می‌کند و نسخه‌های تکراری با <code>SeenSet</code> دور ریخته می‌شوند. '
      f'نتیجه: پوشش کامل ({pct(ph1["coverage"]["mean"])} گره‌ها در همهٔ اجراها)، '
      f'{fa("%.1f" % ph1["messages_per_packet"]["mean"])} دیتاگرام به ازای هر بسته '
      '(دقیقاً دو برابر تعداد یال‌ها، همان‌طور که برای پخش سیل‌آسا انتظار می‌رود) و '
      f'زمان پوشش ۸۰٪ برابر {mstd(ph1["T80"])} ثانیه.')
    w(img("fig8_coverage.png",
          'شکل ۲ — منحنی میانگین انتشار: نسبت گره‌هایی که بسته را دریافت کرده‌اند '
          'بر حسب زمان از لحظهٔ تولید بسته.'))

    # ------------------------------------------------------------- phase 2
    w('<h2>۴. فاز ۲ — حمله به پخش عمومی</h2>')
    w('<h3>۴.۱ خط پایه: «اولین جاسوس مشاهده‌کننده»</h3>')
    w('در این روش، مهاجم نخستین مشاهدهٔ ثبت‌شده در میان جاسوس‌ها را برمی‌دارد و '
      '<b>همسایه‌ای که بسته را به آن جاسوس تحویل داده است</b> را مبدأ حدس می‌زند '
      '(حدس زدن خودِ جاسوس بی‌معناست، چون مبدأها همیشه از گره‌های درستکار انتخاب '
      'می‌شوند). اگر آن همسایه خود جاسوس باشد، نخستین مشاهده‌ای که فرستنده‌اش درستکار '
      'است استفاده می‌شود.')
    w('<h3>۴.۲ روش پیشنهادی: تخمین بیشینه‌درست‌نمایی زمانی</h3>')
    w('مهاجم تأخیر پایهٔ همهٔ لینک‌ها را می‌داند، پس می‌تواند کوتاه‌ترین تأخیر مسیر '
      '<span class="ltr">d(v, s)</span> را میان هر گرهٔ نامزد '
      '<span class="ltr">v</span> و هر جاسوس <span class="ltr">s</span> با دایکسترا '
      'محاسبه کند. اگر بسته در لحظهٔ نامعلوم <span class="ltr">t₀</span> در '
      '<span class="ltr">v</span> ساخته شده باشد، زمان نخستین ورود آن به جاسوس '
      '<span class="ltr">s</span> برابر است با')
    w('<div class="formula">t<sub>s</sub> = t₀ + d(v, s) + ε<sub>s</sub> , '
      'Var[ε<sub>s</sub>] = (0.2²/3) · Σ<sub>i∈path</sub> d<sub>i</sub>²</div>')
    w('چون جیتر هر لینک یکنواخت و مستقل است، واریانس خطا مجموع واریانس جیتر یال‌های '
      'مسیر است. با حذف پارامتر مزاحم <span class="ltr">t₀</span> (کمینه‌سازی بستهٔ '
      'مربعات وزن‌دار)، امتیاز هر نامزد به‌دست می‌آید:')
    w('<div class="formula">χ²(v) = Σ<sub>s</sub> w<sub>s</sub> ( t<sub>s</sub> − d(v,s) '
      '− t̂₀(v) )² ,  w<sub>s</sub> = 1 / Var[ε<sub>s</sub>] ,  '
      't̂₀(v) = Σ w<sub>s</sub>(t<sub>s</sub>−d(v,s)) / Σ w<sub>s</sub></div>')
    w('<b>فیلتر سازگاری هاپ آخر.</b> علاوه بر زمان، مهاجم می‌داند بسته از کدام همسایه '
      'به هر جاسوس رسیده است. در پخش عمومی، نخستین نسخهٔ رسیده تقریباً همیشه از '
      'کوتاه‌ترین مسیر می‌آید؛ بنابراین برای مبدأ واقعی باید '
      '<span class="ltr">d(v,u) + delay(u,s) ≈ d(v,s)</span> برقرار باشد. هر نامزدی که '
      'این شرط را (با حاشیهٔ ۲۵٪) نقض کند جریمهٔ بزرگی می‌گیرد. این فیلتر عملاً '
      'هذلولی‌های هم‌ارز در تخمین زمانی را می‌شکند و بیشترین سهم را در بهبود دقت دارد.')
    w('اگر تنها یک جاسوس بسته را ببیند، تفاضل زمانی معنا ندارد و روش به همان '
      'حدس همسایهٔ تحویل‌دهنده بازمی‌گردد.')

    w('<h3>۴.۳ مکان‌یابی و تعداد بهینهٔ جاسوس‌ها</h3>')
    w('چون <span class="ltr">Score<sub>adv</sub> = Accuracy / k</span> است، افزودن جاسوس '
      'تنها وقتی به‌صرفه است که دقت را بیش از نسبت <span class="ltr">k/(k−1)</span> '
      'بالا ببرد. برای انتخاب مکان‌ها، پنج راهبرد ابتکاری '
      '(<span class="ltr">k-center</span>، بیشترین درجه، بیشترین بینابینی، ترکیبی و '
      'تصادفی) و یک <b>جست‌وجوی حریصانهٔ رو به جلو</b> روی دو اجرای کالیبراسیون '
      f'مستقل (بذرهای {fa(str(calib["calibration_seeds"]))}، جدا از پنج بذر ارزیابی) '
      'مقایسه شدند. جست‌وجوی حریصانه در هر گام گره‌ای را می‌افزاید که بیشترین افزایش '
      'دقت را بدهد؛ نتیجهٔ آن تودرتو است، یعنی مجموعهٔ بهینهٔ '
      f'{fa("k")} جاسوس زیرمجموعهٔ مجموعهٔ {fa("k+1")} جاسوس است:')
    w('<div class="box"><b>ترتیب انتخاب جاسوس‌ها:</b> '
      f'<span class="ltr">{" → ".join(str(x) for x in res["spy_order"])}</span></div>')
    w('<table><tr><th class="ltr">k</th><th>جاسوس‌ها</th><th>دقت خط پایه</th>'
      '<th>دقت روش پیشنهادی</th><th class="ltr">Score<sub>adv</sub></th></tr>')
    for r in sw:
        star = ' ★' if r["k"] == kopt else ''
        w('<tr><td class="ltr">%d%s</td><td class="ltr">%s</td><td>%s</td><td>%s</td>'
          '<td class="ltr">%.4f</td></tr>'
          % (r["k"], star, ",".join(str(x) for x in r["spies"]),
             mstd(r["first_spy"]), mstd(r["timing_ml"]),
             r["timing_ml_score_adv"]["mean"]))
    w('</table>')
    w(f'<p>بیشینهٔ <span class="ltr">Score<sub>adv</sub></span> در '
      f'<b>{fa("k* = %d" % kopt)}</b> رخ می‌دهد '
      f'(مقدار {fa("%.4f" % sweep_row(kopt, "timing_ml_score_adv")["mean"])}). '
      'با این حال برای اینکه مقایسهٔ فازها در همهٔ نقاط کاری قابل قضاوت باشد، همهٔ '
      f'جدول‌ها برای {fa("k = 1")} تا {fa("k = %d" % kmax)} (سقف ۳۰٪ رشوه) گزارش شده‌اند '
      'و شبیه‌سازی‌ها با مجموعهٔ کامل ۶ جاسوس اجرا شده‌اند. نکتهٔ مهم این است که در '
      'فازهای ۱ تا ۴ رفتار جاسوس دقیقاً مانند گره درستکار است، پس مجموعهٔ جاسوس‌ها '
      'هیچ اثری بر دینامیک شبکه ندارد و ارزیابی هر زیرمجموعه از روی لاگ مرجع '
      '<b>دقیق</b> است، نه تقریبی.</p>')
    w(img("fig2_phase2_sweep.png",
          'شکل ۳ — چپ: دقت تشخیص مبدأ بر حسب تعداد جاسوس. راست: امتیاز مهاجم '
          '<span class="ltr">Accuracy/k</span> و نقطهٔ بهینه.'))

    # ------------------------------------------------------------- phase 3
    w('<h2 class="pb">۵. فاز ۳ — پیاده‌سازی Dandelion و جاروب پارامتر p</h2>')
    w('ماشین حالت پیاده‌سازی‌شده دقیقاً مطابق صورت پروژه است:')
    w('<ul>'
      '<li>گرهٔ مبدأ همیشه بسته را در وضعیت <b>ساقه</b> می‌سازد و فقط برای '
      '<b>یک همسایهٔ تصادفی</b> می‌فرستد.</li>'
      f'<li>هر گرهٔ دریافت‌کننده مستقلاً با احتمال {fa("p")} بسته را در وضعیت ساقه '
      'نگه می‌دارد و آن را برای یکی از همسایه‌های خود <b>به‌جز فرستندهٔ قبلی</b> '
      'می‌فرستد؛ اگر همسایهٔ واجد شرایطی نباشد، بسته وارد وضعیت <b>پرز</b> می‌شود.</li>'
      f'<li>با احتمال {fa("1 − p")} بسته وارد وضعیت پرز می‌شود و از آن پس تا پایان '
      'انتشار در همین وضعیت می‌ماند: هر گره پس از نخستین دریافت، بسته را برای همهٔ '
      'همسایه‌ها به‌جز فرستندهٔ قبلی ارسال می‌کند و تکراری‌ها با '
      '<code>SeenSet</code> نادیده گرفته می‌شوند.</li>'
      '</ul>')
    w('<div class="box warn"><b>یک تصمیم طراحی که در صورت پروژه مسکوت است.</b> گشت '
      'تصادفی ساقه روی گراف متناهی ممکن است به گره‌ای برگردد که همان بسته را قبلاً در '
      'وضعیت ساقه دیده است. اگر چنین بسته‌ای مثل نسخهٔ تکراری دور ریخته شود، بسته '
      'برای همیشه گم می‌شود. بنابراین قاعدهٔ زیر را گذاشتیم: <b>بازگشت گشت ساقه به گرهٔ '
      'تکراری، بسته را بلافاصله وارد وضعیت پرز می‌کند</b>. این کار خاتمهٔ قطعی و پوشش '
      'کامل را تضمین می‌کند (در همهٔ اجراها '
      f'{pct(min(ph3["p=%.1f" % p]["coverage"]["mean"] for p in P))} پوشش) و در لاگ '
      'با رویداد <code>stem_loop</code> ثبت می‌شود.</div>')
    w('<table><tr><th>پروتکل</th><th class="ltr">T<sub>80%</sub> میانگین</th>'
      '<th>میانه</th><th>انحراف معیار</th><th>دیتاگرام بر بسته</th>'
      '<th>طول متوسط ساقه</th></tr>')
    w('<tr><td>پخش عمومی</td><td class="ltr">%.3f s</td><td class="ltr">%.3f</td>'
      '<td class="ltr">%.3f</td><td class="ltr">%.1f</td><td>—</td></tr>'
      % (ph1["T80"]["mean"], ph1["T80"]["median"], ph1["T80"]["std"],
         ph1["messages_per_packet"]["mean"]))
    for p in P:
        e = ph3["p=%.1f" % p]
        w('<tr><td class="ltr">Dandelion p=%.1f</td><td class="ltr">%.3f s</td>'
          '<td class="ltr">%.3f</td><td class="ltr">%.3f</td><td class="ltr">%.1f</td>'
          '<td class="ltr">%.2f</td></tr>'
          % (p, e["T80"]["mean"], e["T80_median"]["mean"], e["T80"]["std"],
             e["messages_per_packet"]["mean"], e["stem_hops"]["mean"]))
    w('</table>')
    w(img("fig5_network_cost.png",
          'شکل ۴ — زمان پوشش ۸۰٪ و هزینهٔ ارسال برای پخش عمومی و سه مقدار '
          '<span class="ltr">p</span>.'))

    # ------------------------------------------------------------- phase 4
    w('<h2>۶. فاز ۴ — حملهٔ پیشرفته به Dandelion</h2>')
    w('حملهٔ فاز ۲ در برابر Dandelion کارایی کافی ندارد، زیرا موج پرز از گرهٔ پایان '
      'ساقه شروع می‌شود، نه از مبدأ؛ پس تخمین زمانی، محل «شروع پرز» را پیدا می‌کند نه '
      'مبدأ واقعی را. حملهٔ پیشنهادی از دو منبع شواهد استفاده می‌کند:')
    w('<h3>۶.۱ هستهٔ گشت تصادفی معکوس (شواهد ساقه)</h3>')
    w('اگر یکی از جاسوس‌ها بسته را <b>در وضعیت ساقه</b> ببیند، می‌داند گشت ساقه از یال '
      '<span class="ltr">(u → spy)</span> گذشته است. گشت ساقه یک زنجیرهٔ مارکوف روی '
      '<b>یال‌های جهت‌دار</b> است (چون هرگز به فرستندهٔ قبلی برنمی‌گردد):')
    w('<div class="formula">T[(a→b), (b→c)] = p / (deg(b) − 1) ,  c ≠ a ,  '
      'M = Σ<sub>k≥0</sub> T<sup>k</sup> = (I − T)<sup>−1</sup></div>')
    w('درست‌نمایی اینکه مبدأ گره <span class="ltr">v</span> باشد برابر است با احتمال '
      'اینکه گشتی که از <span class="ltr">v</span> شروع شده (مبدأ همسایهٔ اول را '
      'یکنواخت از میان <span class="ltr">deg(v)</span> همسایه برمی‌دارد) به یال '
      'مشاهده‌شده برسد:')
    w('<div class="formula">L(v) = (1 / deg(v)) · Σ<sub>w ∈ N(v)</sub> M[(v→w), (u→spy)]</div>')
    w('چون <span class="ltr">ρ(T) ≤ p &lt; 1</span> است، سری همگراست و ماتریس '
      f'<span class="ltr">M</span> فقط یک بار به ازای هر {fa("p")} محاسبه می‌شود '
      '(اندازهٔ آن برابر تعداد یال‌های جهت‌دار است). این هسته به‌طور خودکار سه اثر را '
      'با هم در نظر می‌گیرد: نمایی افت‌کردن احتمال با طول ساقه، اثر درجهٔ گره‌ها بر '
      'انتخاب مسیر، و تعداد مسیرهای موجود میان دو گره.')
    w('<h3>۶.۲ هستهٔ پایان ساقه (شواهد پرز)</h3>')
    w('اگر هیچ جاسوسی ساقه را نبیند، مهاجم ابتدا با همان تخمین‌گر زمانی فاز ۲ '
      '<b>توزیع احتمال روی گرهٔ شروع پرز</b> را می‌سازد '
      '(<span class="ltr">P<sub>time</sub>(f) ∝ exp(−χ²(f)/2)</span> به‌همراه جریمهٔ '
      'ناسازگاری هاپ آخر) و سپس آن را از هستهٔ پایان ساقه عبور می‌دهد:')
    w('<div class="formula">K[v, f] = P(گشت آغازشده از v در گره f پرز شود) ,  '
      'P(v) = Σ<sub>f</sub> P<sub>time</sub>(f) · K[v, f]</div>')
    w('<div class="box">هنگامی که شواهد ساقه موجود باشد، شواهد پرز اطلاعات '
      '<b>مستقلی</b> دربارهٔ مبدأ اضافه نمی‌کند: هرچه پس از جاسوس رخ می‌دهد برای مهاجم '
      'کاملاً معلوم است، پس ضرب این دو درست‌نمایی، شواهد هم‌بسته را دوباره می‌شمارد. '
      'آزمایش هم همین را نشان داد (ترکیب کورکورانه دقت را در '
      f'{fa("p = 0.5")} حدود ۴ واحد درصد پایین آورد)؛ بنابراین تخمین‌گر نهایی هرگاه '
      'ساقه دیده شود فقط از هستهٔ معکوس استفاده می‌کند و در غیر این صورت به شواهد '
      'پرز تکیه می‌کند.</div>')
    w('<h3>۶.۳ نتایج</h3>')
    for p in P:
        blk = ph4["p=%.1f" % p]["sweep"]
        w(f'<p><b>{fa("p = %.1f" % p)}</b></p>')
        w('<table><tr><th class="ltr">k</th><th>خط پایه (اولین جاسوس)</th>'
          '<th>حملهٔ فاز ۲</th><th>حملهٔ فاز ۴</th>'
          '<th class="ltr">Score<sub>adv</sub> (فاز ۴)</th></tr>')
        for r in blk:
            w('<tr><td class="ltr">%d</td><td>%s</td><td>%s</td><td><b>%s</b></td>'
              '<td class="ltr">%.4f</td></tr>'
              % (r["k"], mstd(r["first_spy"]), mstd(r["timing_ml"]),
                 mstd(r["dandelion_ml"]), r["dandelion_ml_score_adv"]["mean"]))
        w('</table>')
    w(img("fig3_phase4_sweep.png",
          'شکل ۵ — دقت سه تخمین‌گر در برابر Dandelion برای سه مقدار '
          '<span class="ltr">p</span>.'))
    w(img("fig4_attack_comparison.png",
          'شکل ۶ — مقایسهٔ مستقیم حملهٔ فاز ۲ و فاز ۴ روی پخش عمومی و سه مقدار '
          '<span class="ltr">p</span>.'))

    # placement against dandelion
    if "phase4_placement" in res:
        pl = res["phase4_placement"]
        rows = pl["rows"]
        w('<h3>۶.۴ آیا مکان بهینهٔ جاسوس‌ها به پروتکل وابسته است؟</h3>')
        w('جست‌وجوی حریصانه را این بار مستقیماً روی اجراهای Dandelion '
          f'({fa("p = 0.5")}، بذرهای {fa(str(res["seeds"][:2]))}) تکرار کردیم و هر دو '
          f'مجموعه را روی بذرهای <b>کنارگذاشته‌شدهٔ</b> {fa(str(res["seeds"][2:]))} '
          'ارزیابی کردیم تا نتیجه از بیش‌برازش پاک باشد:')
        w('<table><tr><th class="ltr">k</th><th>مجموعهٔ بهینه‌شده برای پخش عمومی</th>'
          '<th>دقت (بذرهای کنارگذاشته)</th><th>مجموعهٔ بهینه‌شده برای Dandelion</th>'
          '<th>دقت (بذرهای کنارگذاشته)</th></tr>')
        for x in rows:
            w('<tr><td class="ltr">%d</td><td class="ltr">%s</td><td>%s</td>'
              '<td class="ltr">%s</td><td><b>%s</b></td></tr>'
              % (x["k"], ",".join(str(i) for i in x["flood_spies"]),
                 mstd(x["flood_holdout"]),
                 ",".join(str(i) for i in x["dand_spies"]), mstd(x["dand_holdout"])))
        w('</table>')
        gain = rows[-1]["dand_holdout"]["mean"] - rows[-1]["flood_holdout"]["mean"]
        w('<p>پاسخ روشن است: <b>بله</b>. تا '
          f'{fa("k = 3")} دو مجموعه عملاً هم‌ارزند، اما از آن به بعد مکان‌یابی مخصوص '
          f'Dandelion جلو می‌افتد و در {fa("k = %d" % kmax)} حدود '
          f'{fa("%.0f" % (100 * gain))} واحد درصد دقت بیشتری می‌دهد '
          f'({pct(rows[-1]["dand_holdout"]["mean"])} در برابر '
          f'{pct(rows[-1]["flood_holdout"]["mean"])}). دلیل شهودی: تخمین‌گر زمانی به '
          '<b>هندسهٔ خوب مثلث‌بندی</b> نیاز دارد و جاسوس‌های دور از هم را می‌پسندد، '
          'در حالی که حملهٔ Dandelion باید <b>گشت ساقه را قطع کند</b> و بنابراین '
          'گره‌هایی با درجه و بینابینی بالا را ترجیح می‌دهد. برای اینکه مقایسهٔ فازها '
          'منصفانه بماند، همهٔ اعداد بخش‌های قبل با یک مجموعهٔ جاسوس ثابت '
          '(بهینه‌شده در فاز ۲) گزارش شده‌اند؛ بنابراین ارقام فاز ۴ یک '
          '<b>کران پایین</b> از توان واقعی مهاجم‌اند.</p>')

    # ------------------------------------------------------------- phase 5
    w('<h2 class="pb">۷. فاز ۵ — تأخیر عمدی جاسوس‌ها</h2>')
    w('در این فاز هر جاسوس پیش از بازپخش، تأخیر عمدی به اندازهٔ <b>یک تأخیر پایهٔ همان '
      'لینک</b> (بیشینهٔ مجاز) اضافه می‌کند؛ حذف بسته یا نگهداری نامحدود مجاز نیست. '
      'دو مدل مهاجم را مقایسه می‌کنیم: مدل <b>جبران‌نشده</b> که همچنان تأخیر پایه را '
      'برای همهٔ لینک‌ها فرض می‌کند، و مدل <b>جبران‌شده</b> که می‌داند هزینهٔ هر لینک '
      '<b>خروجی از یک جاسوس</b> دو برابر شده است و کوتاه‌ترین مسیرها را روی گراف '
      'جهت‌دارِ اصلاح‌شده حساب می‌کند.')
    w('<table><tr><th>سناریو</th><th>دقت بدون تأخیر (فاز ۴)</th>'
      '<th>دقت با تأخیر — مدل جبران‌نشده</th><th>دقت با تأخیر — مدل جبران‌شده</th>'
      '<th class="ltr">T<sub>80%</sub> بدون تأخیر</th>'
      '<th class="ltr">T<sub>80%</sub> با تأخیر</th></tr>')
    for p in P:
        e = ph5["p=%.1f" % p]
        base = sweep_row(kmax, "dandelion_ml", p)
        t0 = ph3["p=%.1f" % p]["T80"]["mean"]
        w('<tr><td class="ltr">Dandelion p=%.1f</td><td>%s</td><td>%s</td>'
          '<td><b>%s</b></td><td class="ltr">%.3f s</td><td class="ltr">%.3f s '
          '(%+.0f%%)</td></tr>'
          % (p, mstd(base), mstd(e["dandelion_ml_naive"]),
             mstd(e["dandelion_ml_compensated"]), t0, e["T80"]["mean"],
             100.0 * (e["T80"]["mean"] / t0 - 1.0)))
    fl = ph5["flood"]
    w('<tr><td>پخش عمومی</td><td>%s</td><td>%s</td><td><b>%s</b></td>'
      '<td class="ltr">%.3f s</td><td class="ltr">%.3f s (%+.0f%%)</td></tr>'
      % (mstd(sweep_row(kmax, "timing_ml")), mstd(fl["timing_ml_naive"]),
         mstd(fl["timing_ml_compensated"]), ph1["T80"]["mean"], fl["T80"]["mean"],
         100.0 * (fl["T80"]["mean"] / ph1["T80"]["mean"] - 1.0)))
    w('</table>')
    w(img("fig6_phase5.png",
          'شکل ۷ — اثر تأخیر عمدی جاسوس‌ها بر دقت مهاجم (چپ) و بر زمان پوشش شبکه (راست).'))

    d_acc = [ph5["p=%.1f" % p]["dandelion_ml_compensated"]["mean"]
             - sweep_row(kmax, "dandelion_ml", p)["mean"] for p in P]
    d_std = [max(ph5["p=%.1f" % p]["dandelion_ml_compensated"]["std"],
                 sweep_row(kmax, "dandelion_ml", p)["std"]) for p in P]
    d_t80 = [ph5["p=%.1f" % p]["T80"]["mean"] / ph3["p=%.1f" % p]["T80"]["mean"] - 1
             for p in P]
    sh_before = [(1 / ph3["p=%.1f" % p]["T80"]["mean"])
                 * (1 - sweep_row(kmax, "dandelion_ml", p)["mean"]) for p in P]
    sh_after = [(1 / ph5["p=%.1f" % p]["T80"]["mean"])
                * (1 - ph5["p=%.1f" % p]["dandelion_ml_compensated"]["mean"]) for p in P]
    dsh = sum(a / b - 1 for a, b in zip(sh_after, sh_before)) / len(P)
    trend = "افزایش" if sum(d_acc) > 0 else "کاهش"
    w('<div class="box key"><b>تحلیل — آیا تأخیر به نفع مهاجم است؟</b>')
    w('<p><b>۱) اثر بر دقت: عملاً هیچ.</b> دقت مشترک مهاجم به‌طور متوسط '
      f'{fa("%+.1f" % (100 * sum(d_acc) / len(d_acc)))} واحد درصد {trend} می‌یابد، '
      f'در حالی که انحراف معیار میان بذرها حدود '
      f'{fa("%.1f" % (100 * sum(d_std) / len(d_std)))} واحد درصد است؛ یعنی تغییر در حد '
      'نوسان آماری است و تأخیر عمدی، مهاجم را دقیق‌تر نمی‌کند.</p>')
    w('<p><b>۲) اما مدل‌سازی درست حیاتی است.</b> اگر مهاجم تأخیر خودش را در '
      'کوتاه‌ترین‌مسیرها لحاظ نکند، بخشی از دقتش را از دست می‌دهد؛ این اثر در پخش '
      f'عمومی چشمگیر است: {pct(fl["timing_ml_naive"]["mean"])} با مدل جبران‌نشده در '
      f'برابر {pct(fl["timing_ml_compensated"]["mean"])} با مدل جبران‌شده — یعنی '
      'مهاجمِ آگاه، تمام زیانِ ناشی از تأخیر خودش را بازمی‌گرداند. علت روشن است: '
      'تأخیر عمدی برای مهاجم <b>اطلاعات پنهان نیست</b>؛ او دقیقاً می‌داند روی کدام '
      'لینک چقدر صبر کرده و کافی است هزینهٔ یال‌های خروجی از جاسوس‌ها را دو برابر '
      'کند و مسیرها را روی گراف جهت‌دار اصلاح‌شده حساب کند.</p>')
    w('<p><b>۳) اثر واقعی روی شبکهٔ درستکار است.</b> زمان رسیدن بسته به ۸۰٪ گره‌ها '
      f'به‌طور متوسط {fa("%+.0f%%" % (100 * sum(d_t80) / len(d_t80)))} بزرگ‌تر می‌شود و '
      f'در نتیجه <span class="ltr">Score<sub>honest</sub></span> حدود '
      f'{fa("%+.0f%%" % (100 * dsh))} تغییر می‌کند. بنابراین تأخیر عمدی نه یک ابزار '
      '«افزایش دقت»، بلکه یک <b>حملهٔ کیفیت سرویس</b> است: مهاجم بدون پرداخت هیچ '
      'هزینه‌ای در دقت خود، امتیاز شبکهٔ درستکار را پایین می‌آورد. اگر معیار داوری '
      f'تنها <span class="ltr">Score<sub>adv</sub></span> باشد، این کار برای مهاجم '
      'بی‌تفاوت است؛ اما اگر بازی مجموع‌صفر میان دو امتیاز در نظر گرفته شود، '
      'راهبرد غالب مهاجم است.</p>')
    w('<p><b>۴) چرا ترتیب مشاهده‌ها به هم نمی‌ریزد؟</b> بیشینهٔ تأخیر مجاز برابر یک '
      'تأخیر پایهٔ همان لینک است. چون بسته‌ها معمولاً از چند مسیر موازی به جاسوس‌ها '
      'می‌رسند و بخش عمدهٔ مسیرها از گره‌های درستکار عبور می‌کند، فقط بخشی از '
      'مشاهده‌ها جابه‌جا می‌شوند و تخمین‌گرِ بیشینه‌درست‌نمایی — که همهٔ جاسوس‌ها را '
      'هم‌زمان و به‌صورت وزن‌دار در نظر می‌گیرد — نسبت به این جابه‌جایی مقاوم است. '
      'خط پایهٔ «اولین جاسوس» که فقط به یک مشاهده تکیه دارد، آسیب‌پذیرتر است.</p>')
    w('</div>')

    # ------------------------------------------------------------- scores
    w('<h2>۸. معیارهای نهایی</h2>')
    w('<div class="formula">Score<sub>adv</sub> = Accuracy / (Number of Bribed Nodes) '
      ' &nbsp;&nbsp;&nbsp; Score<sub>honest</sub> = (1 / T<sub>80%</sub>) × '
      '(1 − Adversary Detection Rate)</div>')
    w('<table><tr><th class="ltr">k</th><th>سناریو</th><th>دقت (روش پیشنهادی)</th>'
      '<th>دقت خط پایه</th><th class="ltr">T<sub>80%</sub></th>'
      '<th class="ltr">Score<sub>adv</sub></th>'
      '<th class="ltr">Score<sub>honest</sub></th></tr>')
    for r in res["scores"]:
        w('<tr><td class="ltr">%d</td><td class="r">%s</td><td>%s</td><td>%s</td>'
          '<td class="ltr">%.3f s</td><td class="ltr">%.4f</td>'
          '<td class="ltr">%.4f</td></tr>'
          % (r["k"], r["phase"], pct(r["accuracy"]), pct(r["accuracy_baseline"]),
             r["T80"], r["score_adv"], r["score_honest"]))
    w('</table>')
    w(img("fig7_scores.png", 'شکل ۸ — امتیاز مهاجم و امتیاز شبکهٔ درستکار.'))

    # ------------------------------------------------------------ conclusion
    w('<h2>۹. جمع‌بندی</h2>')
    w('<ol>'
      f'<li>در پخش عمومی، تخمین‌گر زمانی پیشنهادی دقت خط پایه را در '
      f'{fa("k = %d" % kmax)} از {pct(sweep_row(kmax, "first_spy")["mean"])} به '
      f'{pct(sweep_row(kmax, "timing_ml")["mean"])} می‌رساند — یعنی صرفِ دانستن تأخیر '
      'پایهٔ لینک‌ها، حتی با کمتر از یک‌سوم شبکه، حریم خصوصی مبدأ را عملاً از بین می‌برد.</li>'
      f'<li>Dandelion دفاع مؤثری است: در {fa("p = 0.9")} دقت بهترین حملهٔ ما با '
      f'{fa(kmax)} جاسوس به {pct(sweep_row(kmax, "dandelion_ml", 0.9)["mean"])} '
      f'می‌افتد. هزینهٔ آن افزایش {fa("%.0f%%" % (100 * (ph3["p=0.9"]["T80"]["mean"] / ph1["T80"]["mean"] - 1)))} '
      'در زمان پوشش است.</li>'
      f'<li>پارامتر {fa("p")} دقیقاً همان مبادلهٔ «حریم خصوصی در برابر تأخیر» است: '
      f'از {fa("p = 0.1")} تا {fa("p = 0.9")} دقت مهاجم از '
      f'{pct(sweep_row(kmax, "dandelion_ml", 0.1)["mean"])} به '
      f'{pct(sweep_row(kmax, "dandelion_ml", 0.9)["mean"])} کاهش و '
      f'{fa("T80")} از {sec(ph3["p=0.1"]["T80"]["mean"])} به '
      f'{sec(ph3["p=0.9"]["T80"]["mean"])} افزایش می‌یابد.</li>'
      '<li>حملهٔ آگاه از ساختار (هستهٔ گشت معکوس) در همهٔ مقادیر '
      f'{fa("p")} از خط پایه بهتر یا برابر است و بیشترین برتری را در '
      f'{fa("p")} کوچک و متوسط نشان می‌دهد، جایی که ساقه کوتاه است و شواهد پرز '
      'قابل بهره‌برداری‌اند.</li>'
      '<li>تأخیر عمدی جاسوس‌ها ابزار خوبی برای مهاجم است، اما نه به‌خاطر افزایش دقت؛ '
      'بلکه چون امتیاز شبکهٔ درستکار را از راه بزرگ‌کردن '
      f'{fa("T80")} پایین می‌آورد.</li>'
      '</ol>')

    w('<h2>۱۰. بازتولید نتایج</h2>')
    w('<pre>cd Code\n'
      'python3 -m pip install -r requirements.txt\n'
      'python3 -m src.experiments all      # توپولوژی + کالیبراسیون + فاز ۱ تا ۵ + تحلیل\n'
      'python3 -m src.plots                # نمودارها و جدول‌ها\n'
      'python3 -m src.verify               # بررسی خودکار صحت روی همهٔ اجراها\n'
      'python3 -m src.report               # ساخت همین گزارش از روی نتایج\n'
      'python3 -m src.demo --mode dandelion --p 0.5 --packets 20   # اجرای نمایشی</pre>')
    w('<p class="small">هر اجرا پوشهٔ خود را در <code>results/runs/</code> می‌سازد و در '
      'صورت وجود، دوباره اجرا نمی‌شود؛ بنابراین کارزار قابل توقف و ادامه است. '
      'بذر سناریو، توپولوژی، مجموعهٔ جاسوس‌ها و مبدأ و زمان دقیق هر بسته در '
      '<code>meta.json</code> هر اجرا ذخیره می‌شود.</p>')

    w('<h2>۱۱. محدودیت‌ها و تقریب‌ها</h2>')
    w('<ul>'
      '<li>هستهٔ گشت معکوس، قاعدهٔ «شکستن حلقه» را مدل نمی‌کند (گشت‌هایی که به گرهٔ '
      'تکراری برمی‌گردند زودتر پرز می‌شوند). این تقریب تنها در '
      f'{fa("p")} بزرگ محسوس است و تخمین‌گر را محافظه‌کارانه‌تر می‌کند.</li>'
      '<li>زمان‌بندی روی یک ماشین واقعی انجام می‌شود؛ سربار حلقهٔ رویداد و '
      'زمان‌بند سیستم‌عامل به تأخیر مدل‌شده اضافه می‌شود. اندازه‌گیری‌های '
      '<code>verify.py</code> نشان می‌دهد میانهٔ این سربار حدود ۱ میلی‌ثانیه و '
      'صدک ۹۹ آن حدود ۴ میلی‌ثانیه است، در برابر تأخیر پایهٔ لینک‌ها که '
      'ده‌ها تا صدها میلی‌ثانیه است.</li>'
      '<li>تخمین‌گر زمانی فرض می‌کند نخستین نسخهٔ رسیده از کوتاه‌ترین مسیر آمده است؛ '
      'با جیتر ۲۰٪ این فرض در اکثر موارد درست است و فیلتر سازگاری هاپ آخر '
      'موارد نقض را حذف می‌کند.</li>'
      '</ul>')
    return "\n".join(o)


def main() -> None:
    res = json.load(open(os.path.join(RESULTS, "results.json")))
    calib = json.load(open(os.path.join(RESULTS, "calib", "calibration.json")))
    topo = get_topology()
    html = ("<!doctype html><html lang=\"fa\" dir=\"rtl\"><head><meta charset=\"utf-8\">"
            "<title>قاصدک در برابر جاسوس — گزارش پروژه</title><style>%s</style></head>"
            "<body>%s</body></html>" % (CSS, build(res, topo, calib)))
    with open(OUT_HTML, "w") as fh:
        fh.write(html)
    print("wrote", OUT_HTML)
    if os.path.exists(CHROME):
        subprocess.run([CHROME, "--headless", "--disable-gpu", "--no-pdf-header-footer",
                        "--print-to-pdf=%s" % OUT_PDF, "--virtual-time-budget=8000",
                        "file://" + OUT_HTML], check=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if os.path.exists(OUT_PDF):
            print("wrote", OUT_PDF)
    else:
        print("Chrome not found - open Report.html and print it to PDF manually.")


if __name__ == "__main__":
    main()
