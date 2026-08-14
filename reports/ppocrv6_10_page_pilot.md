# PP-OCRv6 10-page diagnostic pilot

> This is a diagnostic comparison without typed human ground truth. Similarity and confidence are not CER/WER or proof of accuracy.

- Device: `gpu:0`
- Pages completed: **10**
- Total runtime: **12.45 s**
- Mean page runtime: **1.0625 s**
- Mean PP-OCRv6 confidence: **0.973396**
- Vietnamese diacritic marks: **1253** (PP-OCRv6) vs **4009** (current corpus)
- Diacritic ratio vs current corpus: **31.3%**
- Pilot decision: **do not replace the current Vietnamese recognizer**

## Findings

PP-OCRv6 produced cleaner line detection and removed much scan-border noise, but systematically dropped Vietnamese tone marks and occasionally letters on all inspected document types. Its mean recognition confidence remained high, so confidence is not a safe quality gate for this failure mode.

The current Tesseract path is noisier on borders and marginal marks, but preserves substantially more Vietnamese orthography. PP-OCRv6 may still be evaluated later as a detector/layout component, or after Vietnamese-specific recognizer fine-tuning, but this pilot rejects it as a drop-in text recognizer.

| Subject | PDF page | Category | PP chars | Current chars | Similarity | PP confidence | Runtime |
|---|---:|---|---:|---:|---:|---:|---:|
| VNR202 | 22 | chapter_opening | 1762 | 1998 | 0.526 | 0.975 | 1.66 s |
| VNR202 | 125 | chapter_opening | 1546 | 1774 | 0.619 | 0.972 | 1.26 s |
| VNR202 | 181 | body_control | 2580 | 2941 | 0.391 | 0.969 | 1.71 s |
| VNR202 | 226 | late_review | 2496 | 2834 | 0.412 | 0.966 | 1.64 s |
| MLN131 | 47 | low_quality | 198 | 220 | 0.780 | 0.986 | 0.55 s |
| MLN131 | 161 | low_quality_heading | 275 | 313 | 0.581 | 0.985 | 0.61 s |
| MLN131 | 235 | low_quality_heading | 262 | 306 | 0.676 | 0.988 | 0.59 s |
| HCM202 | 72 | ocr_noise_heading | 1288 | 1476 | 0.441 | 0.976 | 1.01 s |
| HCM202 | 84 | ocr_noise_heading | 1214 | 1371 | 0.587 | 0.981 | 0.91 s |
| HCM202 | 267 | late_ocr_noise | 531 | 633 | 0.672 | 0.971 | 0.69 s |

## Text samples

### VNR202 - PDF page 22

**PP-OCRv6**

```text
Chưong 1
ĐĂNG CÔNG SN VIĘT NAM RA DÒI VÀ LÃNH DAO
DÁU TRANH GIÀNH CHÍNH QUYÈN (1930 - 1945)
MUC TIÊU
V kin thc:
Cung cp cho sinh viên nhng tri thc có tính h thng quá trình ra đi ca Đng
Cng sn Vit Nam (1920-1930), ni dung cơ bn, giá tr lch s ca Cương lĩnh chính tr
đu tiên ca Đng và quá trình Đng lãnh đo cuc đu tranh gii phóng dân tc, giành
chính quyèn (1930-1945).
V tư tưng:
Cung cp cơ s lch s, góp phn cng c nim tin ca th h tr vào con đưng
cách mng gii phóng dân tc và phát trin đt nưc-s la chn đúng đn, tt yu, khách
quan ca lnh t Nguyn Ái Quc và Đng Cng sn Vit Nam thi k đu dng Đng.
V k năng:
T vic nhn thc lch s thi k đu dng Đng, góp phn trang b cho sinh viên
phương pháp nhn thc bin chng, khách quan v quá trình Đng ra đi và vai trò lãnh
đo ca Đng trong cuc đu tranh gii phóng dân tc, xác lp chính quyn cách mng.
I. Đng Cng sn Vit Nam ra đi và Cưong līnh chính tr đu tiên ca Đng
(tháng 2- 1930)
1. Bi cnh lch s
T na sau th k XIX, các nưc tư bn Âu-M có nhng chuyn bin mnh m
trong đi sng kinh t-xã hi. Ch nghĩa tư bn phương Tây chuyn nhanh t giai đon t
do cnh tranh sang giai đon đc quyn (giai đon đ quc ch nghĩa), đy mnh quá
trình xâm chim và nô dch các nưc nh, yu  châu Á, châu Phi và khu vc M-Latinh,
bin các quc gia này thành thuc đa ca các nưc đ quc. Trưc bi cnh đó, nhân dân
các dân tc b áp bc đã đng lên đu tranh t gii phóng khi ách thc dân, đ quc, to
thành phong trào gii phóng dân tc mnh m, rng khp, nht là  châu Á. Cùng vi
phong trào đu tranh ca giai cp vô sn chng li giai cp tư sn  các nưc tư bn ch
nghĩa, phong trào gii phóng dân tc  các nưc thuc đa tr thành mt b phn quan
trng trong cuc đu tranh chung chng tư bn, thc dân. Phong trào gii phóng dân tc
các nuc châu Á đu th k XX phát trin rng khp, tác đng mnh m đn phong trào
yêu nưc Vit Nam.
14
8
```

**Current corpus (Tesseract path)**

```text
Chương 1
ĐẢNG CỘNG SẢN VIỆT NAM RA ĐỜI VÀ LÃNH ĐẠO
ĐẦU TRANH GIÀNH CHÍNH QUYỀN (1930 - 1945)
MỤC TIÊU
Về kiến thức:
Cung cấp cho sinh viên những tri thức có tính hệ thống quá trình ra đời của Đảng
Cộng sản Việt Nam (1920-1930), nội dung cơ bản, giá trị lịch sử của Cương lĩnh chính trị
đầu tiên của Đảng và quá trình Đảng lãnh đạo cuộc đấu tranh giải phóng dân tộc, giành
chính quyền (1930-1945).
Về tư tưởng:
Cung cấp cơ sở lịch sử, góp phần củng cố niềm tin của thế hệ trẻ vào con đường
cách mạng giải phóng dân tộc và phát triển đất nước-sự lựa chọn đúng đắn, tất yếu, khách
quan của lãnh tụ Nguyễn Ái Quốc và Đảng Cộng sản Việt Nam thời kỳ đầu dựng Đảng.
Về kỹ năng:
Từ việc nhận thức lịch sử thời kỳ đầu dựng Đảng, góp phần trang bị cho sinh viên
phương pháp nhận thức biện chứng, khách quan về quá trình Đảng ra đời và vai trò lãnh
đạo của Đảng trong cuộc đấu tranh giải phóng dân tộc, xác lập chính quyền cách mạng.
I. Đảng Cộng sản Việt Nam ra đời và Cương lĩnh chính trị đầu tiên của Đáng
(tháng 2- 1930)
1. Bối cảnh lịch sử
Từ nửa sau thế kỷ XIX, các nước tư bản Âu-Mỹ có những chuyển biến mạnh mẽ
trong đời sống kinh tế-xã hội. Chủ nghĩa tư bản phương Tây chuyển nhanh từ giai đoạn tự
do cạnh tranh sang giai đoạn độc quyền (giai đoạn đế quốc chủ nghĩa), đẩy mạnh quá
trình xâm chiếm và nô dịch các nước nhỏ, yếu ở châu Á, châu Phi và khu vực Mỹ-Latinh,
biến các quốc gia này thành thuộc địa của các nước đề quốc. Trước bối cảnh đó, nhân dân
các dân tộc bị áp bức đã đứng lên đấu tranh tự giải phóng khỏi ách thực dân, đề quốc, tạo
thành phong trào giải phóng dân tộc mạnh mẽ, rộng khắp, nhất là ở châu Á. Cùng với
phong trào đấu tranh của giai cấp vô sản chống lại giai cấp tư sản ở các nước tư bản chủ
nghĩa, phong trào giải phóng dân tộc ở các nước thuộc địa trở thành một bộ phận quan

```

### VNR202 - PDF page 125

**PP-OCRv6**

```text
Chưong 3
ĐÃNG LÃNH DAO CÃ NUÓC QUÁ DO LÊN
CHÙ NGHÃA XÃ HOI VÀ TIÉN HÀNH CÔNG CUOC DOI MÓI
(1975 - 2018)
MUC TIÊU
V kin thúc:
Giúp sinh viên nm đưc đưng li, Cương lĩnh, nhng tri thc có h thng v quá
trình phát trin đưng li và lãnh đo ca Đng đưa c nưc quá đ lên ch nghĩa xã hi
và tin hành công cuc đi mi tù sau ngày thng nht đt nưc năm 1975 đn nay.
V tư tưng:
Cng c nim tin ca sinh viên v nhng thng li ca Đng trong lãnh đo đưa c
nưóc quá đ xây dưng ch nghĩa xã hi và tin hành công cuc đi mói (1975-2018),
cng c nim tin và lòng t hào vào s lãnh đo ca Đng đi vi s nghip cách mng
hin nay.
V k năng:
Rèn luyn cho hc viên phong cách tư duy lý lun gn lin vi thc tin, phát huy
tính năng đng, sáng to ca ngưi hc; vn dng nhng tri thc v s lãnh đo ca Đng
vào thc tin cuc sng.
I. Lãnh đo c nuưóc xây dng ch nghĩa xã hi và bo v T quc (1975-1986)
1. Xây dng ch nghĩa xã hi và bo v T quc 1975-1981
Hoàn cnh lch s ca thi k sau năm 1975 là đt nưc đã hòa bình, đc lp, thng
nht, c nưc quá đ lên ch nghĩa xã hi. Đt nưc có nhiu thun li vi sc mnh tng
hp, đng thi cũng phi khc phc nhng hu qu nng n ca chin tranh. Đim xut
phát ca Vit Nam v kinh t- xã hi còn  trình đ thp. Điu kin quc t có thun li
đồng thi có xut hin nhng khó khăn thách thc mi. Các nưc xã hi ch nghĩa bc l
nhng khó khăn v kinh t - xã hi và s phát trin; các th lc thù đch bao vây cm vn
và phá hoi s phát trin ca Vit Nam.
Hoàn thành thng nht đt nưc v mt nhà nuc
Sau đi thng mùa Xuân năm 1975, đt nưc ta bưc vào mt k nguyên mi: T
quc hoàn toàn đc lp, thng nht, quá đ đi lên ch nghĩa xã hi. Đ thc hin bưc
8
117
```

**Current corpus (Tesseract path)**

```text
Chương 3
ĐẢNG LÃNH ĐẠO CẢ NƯỚC QUÁ ĐỘ LÊN
| CHỦ NGHĨA XÃ HỘI VÀ TIỀN HÀNH CÔNG CUỘC ĐÓI MỚI
(1975 -2018)
| MỤC TIÊU
| Về kiến thức:
Giúp sinh viên nắm được đường lối, Cương lĩnh, những tri thức có hệ thống về quá
| trình phát triển đường lối và lãnh đạo của Đảng đưa cả nước quá độ lên chủ nghĩa xã hội
| và tiền hành công cuộc đồi mới từ sau ngày thống nhất đất nước năm 1975 đến nay.
Về tư tưởng:
| Củng cố niềm tin của sinh viên về những thắng lợi của Đảng trong lãnh đạo đưa cả
nước quá độ xây dưng chủ nghĩa xã hội và tiến hành công cuộc đổi mới (1975-2018),
củng cố niềm tin và lòng tự hào vào sự lãnh đạo của Đảng đối với sự nghiệp cách mạng
hiện nay.
về kỹ năng:
Rèn luyện cho học viên phong cách tư duy lý luận gắn liền với thực tiễn, phát huy
tính năng động, sáng tạo của người học; vận dụng những tri thức về sự lãnh đạo của Đảng
vào thực tiễn cuộc sống.
I Lãnh đạo cả nước xây dựng chủ nghĩa xã hội và bảo vệ Tổ quốc (1975-1986)
1. Xây dựng chủ nghĩa xã hội và bảo vệ Tổ quốc 1975-1981
Hoàn cảnh lịch sử của thời kỳ sau năm 1975 là đất nước đã hòa bình, độc lập, thống
nhất, cả nước quá độ lên chủ nghĩa xã hội. Đất nước có nhiều thuận lợi với sức mạnh tổng
hợp, đồng thời cũng phải khắc phục những hậu quả nặng nề của chiến tranh. Điểm xuất
phát của Việt Nam về kinh tế- xã hội còn ở trình độ thấp. Điều kiện quốc tế có thuận lợi
đồng thời có xuất hiện những khó khăn thách thức mới. Các nước xã hội chủ nghĩa bộc lộ
những khó khăn về kinh tế - xã hội và sự phát triển; các thế lực thù địch bao vây cấm vận
và phá hoại sự phát triển của Việt Nam.
Hoàn thành thống nhất đắt nước về mặt nhà nước
Sau đại thắng mùa Xuân năm 1975, đất nước ta bước vào một kỷ nguyên mới: Tổ
quốc hoàn toàn độc lập, thống nhất, quá độ đi lên chủ nghĩa xã hội. Để thực hiện bước
117 s
```

### VNR202 - PDF page 181

**PP-OCRv6**

```text
Cương lĩnh năm 2011 chí rõ tám phương hưóng cơ bn xây dng ch nghĩa xã hi
nưc ta: Mt là, đy mnh công nghip hoá, hin đi hoá đt nưc gn vi phát trin kinh
t tri thc, bo v tài nguyên, môi trưng. Hai là, phát trin nn kinh t th trưòng đnh
hưóng xã hi ch nghĩa. Ba là, xây dng nn vǎn hoá tiên tin, đm đà bn sc dân tc;
xây dng con ngưi, nâng cao đi sng nhân dân, thc hin tin b và công bng xã hi.
Bn là, bo đm vng chc quc phòng và an ninh quc gia, trt t, an toàn xã hi. Năm
là, thc hin đưòng li đi ngoi đc lp, t ch, hoà bình, hu ngh, hp tác và phát
trin; ch đng và tích cc hi nhp quc t. Sáu là, xây dng nn dân ch xã hi ch
nghĩa, thc hin đi đoàn kt toàn dân tc, tăng cưng và m rng mt trn dân tc thng
nht. By là, xây dng Nhà nưc pháp quyn xã hi ch nghĩa ca nhân dân, do nhân dân,
vì nhân dân. Tám là, xây dng Đng trong sch, vng mnh.
Cương lĩnh năm 2011 b sung cn nm vng và gii quyt tt tám mi quan h lón:
Quan h gia đi mi, n đnh và phát trin; gia đi mi kinh t và đi mi chính tr;
gia kinh t th trưng và đnh hưóng xã hi ch nghĩa; gia phát trin lc lưng sn xut
và xây dng, hoàn thin tùng bưc quan h sn xut xã hi ch nghĩa; gia tăng trưng
kinh t và phát trin văn hoá, thc hin tin b và công bng xã hi; gia xây dng ch
nghĩa xã hi và bo v T quc xã hi ch nghĩa; gia đc lp, t ch và hi nhp quc
t; gia Đng lãnh đo, Nhà nưc qun lý, nhân dân làm ch.
Nhng đnh hưóng lón v phát trin kinh t, văn hóa, xã hi, quc phòng, an ninh,
đi ngoi
Phát trin nn kinh t th trưng đnh hưóng xã hi ch nghĩa vi nhiu hình thc s
hu, nhiu thành phn kinh t, hình thc t chúc kinh doanh và hình thc phân phi. Các
thành phn kinh t hot đng theo pháp lut đu là b phn hp thành quan trng ca nn
kinh t, bình đng trưóc pháp lut, cùng phát trin lâu dài, hp tác và cnh tranh lành
mnh. Kinh t nhà nưóc gi vai trò ch đo. Kinh t tp th không ngng đưc cng c
và phát trin
```

**Current corpus (Tesseract path)**

```text
Cương lĩnh năm 2011 chỉ rõ tám phương hướng cơ bản xây dựng chủ nghĩa xã hội ở
nước ta: Một là, đầy mạnh công nghiệp hoá, hiện đại hoá đất nước gắn với phát triển kinh
tế tri thức, bảo vệ tài nguyên, môi trường. //ai là, phát triển nền kinh tế thị trường định
hướng xã hội chủ nghĩa. 8a là, xây dựng nền văn hoá tiên tiến, đậm đà bản sắc dân tộc;
xây dựng con người, nâng cao đời sống nhân dân, thực hiện tiền bộ và công bằng xã hội.
Bồn là, bảo đảm vững chắc quốc phòng và an ninh quốc gia, trật tự, an toàn xã hội. Văm
là, thực hiện đường lối đối ngoại độc lập, tự chủ, hoà bình, hữu nghị, hợp tác và phát
| triển; chủ động và tích cực hội nhập quốc tế. Sáu là, xây dựng nền dân chủ xã hội chủ
nghĩa, thực hiện đại đoàn kết toàn dân tộc, tăng cường và mở rộng mặt trận dân tộc thống
| nhất. Bảy là, xây dựng Nhà nước pháp quyền xã hội chủ nghĩa của nhân dân, do nhân dân,
vì nhân dân. 7ám là, xây dựng Đảng trong sạch, vũng mạnh.
Cương lĩnh năm 2011 bổ sung cần nắm vững và giải quyết tốt tám mối quan hệ lớn:
Quan hệ giữa đổi mới, ổn định và phát triển; giữa đổi mới kinh tế và đổi mới chính trị;
giữa kinh tế thị trường và định hướng xã hội chủ nghĩa; giữa phát triển lực lượng sản xuất
và xây dựng, hoàn thiện từng bước quan hệ sản xuất xã hội chủ nghĩa; giữa tăng trưởng
| kinh tế và phát triển văn hoá, thực hiện tiến bộ và công bằng xã hội; giữa xây dựng chủ
nghĩa xã hội và bảo vệ Tổ quốc xã hội chủ nghĩa; giữa độc lập, tự chủ và hội nhập quốc
tế; giữa Đảng lãnh đạo, Nhà nước quản lý, nhân dân làm chủ.
Những định hướng lớn về phát triển kinh tế, văn hóa, xã hội, quốc phòng, an ninh,
đồi ngoại
Phát triển nền kinh tế thị trường định hướng xã hội chủ nghĩa với nhiều hình thức sở
hữu, nhiều thành phần kinh tế, hình thức tổ chức kinh doanh và hình thức phân phối. Các
| thành phần kinh tế 
```

### VNR202 - PDF page 226

**PP-OCRv6**

```text
Đoàn kt là nguyên tc ca Đng chân chính cách mng. Trong Tuyên ngôn ca
Đng Cng sn (1848), Karl Marx và Friedrich Engels đã nêu rã khu hiu chin lưc:
V sn tt c các nưc đoàn kt li. Đu th k XX, V.I.Lenin và Quc t Cng sn b
sung: Vô sn toàn th gii và các dân tc b áp bc đoàn kt li. Đi vi dân tc Vit
Nam, đoàn kt là truyn thng quý báu, là ci nguồn súc mnh trong s nghip dng
nưóc và gi nưc. Hồ Chí Minh đc bit chú trng nêu cao ngn cò dân tc, li ích quc
gia, dân tc. Đi đoàn kt dân tc là đim căn bn và ni bt trong tư tưng H Chí Minh.
Đoàn kt là sc mnh, đoàn kt là thành công. Ngưi coi gi gìn đoàn kt trong Đng như
gi gìn con ngươi ca mt mình. Ngưi cũng là hin thân ca tinh thn đoàn kt quc t.
Nh sc mnh đi đoàn kt dân tc, tp hp trong Mt trn dân tc thng nht do
Đng lãnh đo, toàn dân Vit Nam đã to nên sc mnh tng hp, làm nên thành công ca
Cách mng Tháng Tám và các cuc kháng chin cu nưc, tranh th đưc s đoàn kt và
ng h ca đng chí, bè bn và nhân dân th giói, thc hin thành công s nghip đi
mi, xây dng, phát trin đt nưc và bo v vng chc T quc.
Trong công cuc đi mi, Đng nhn mnh chin lưc đi đoàn kt dân tc, ly mc
tiêu chung ca li ích quc gia, dân tc làm đim tương đồng, tôn trng li ích ca các
tng lóp, giai cp không trái vi li ích chung, Khép li quá kh, xóa b đnh kin, hn
thù, mc cm, hưóng ti tưoơng lai. Đi đoàn kt dân tc, nhân dân luôn luôn gn lin vói
phát huy và hoàn thin dân ch xã hi ch nghĩa, phát huy quyn làm ch ca nhân dân,
tôn trng quyn con ngưi, quyn và trách nhim công dân. Trong lch s, Đng đã chú
trng xây dng các t chc Mt trn, các đoàn th đ đoàn kt toàn dân, ngày nay, tăng
cưòng xây dng Mt trn T quc Vit Nam và các t chc chính tr-xã hi thc hin tt
nht đoàn kt mi giai cp, tng lóp, dân tc, tôn giáo, ngưi Vit Nam đnh cư  nưc
ngoài, to s đồng thun xã hi.
4. Kt hp sc mnh dân tc vi sc mnh thi đi, sc mnh tro
```

**Current corpus (Tesseract path)**

```text
Đoàn kết là nguyên tắc của Đảng chân chính cách mạng. Trong Tuyên ngôn của
Đảng Cộng sản (1848), Karl Marx và Friedrich Engels đã nêu rõ khẩu hiệu chiến lược:
Vô sản tắt cả các nước đoàn kết lại. Đầu thế kỷ XX, V.I.Lenin và Quốc tế Cộng sản bổ
sung: Vô sản toàn thế giới và các dân tộc bị áp bức đoàn kết lại. Đối với dân tộc Việt
Nam, đoàn kết là truyền thống quý báu, là cội nguồn sức mạnh trong sự nghiệp dựng
nước và giữ nước. Hồ Chí Minh đặc biệt chú trọng nêu cao ngọn cờ dân tộc, lợi ích quốc
gia, dân tộc. Đại đoàn kết dân tộc là điểm căn bản và nổi bật trong tư tưởng Hồ Chí Minh.
Đoàn kết là sức mạnh, đoàn kết là thành công. Người coi giữ gìn đoàn kết trong Đảng như
giữ gìn con ngươi của mắt mình. Người cũng là hiện thân của tỉnh thần đoàn kết quốc tế.
Nhờ sức mạnh đại đoàn kết dân tộc, tập hợp trong Mặt trận dân tộc thống nhất do
Đảng lãnh đạo, toàn dân Việt Nam đã tạo nên sức mạnh tổng hợp, làm nên thành công của
Cách mạng Tháng Tám và các cuộc kháng chiến cứu nước, tranh thủ được sự đoàn kết và
ủng hộ của đồng chí, bè bạn và nhân dân thế giới, thực hiện thành công sự nghiệp đồi
mới, xây dựng, phát triển đất nước và bảo vệ vững chắc Tổ quốc.
Trong công cuộc đổi mới, Đảng nhấn mạnh chiến lược đại đoàn kết dân tộc, lấy mục
tiêu chung của lợi ích quốc gia, dân tộc làm điểm tương đồng, tôn trọng lợi ích của các
tằng lớp, giai cấp không trái với lợi ích chung, Khép lại quá khứ, xóa bỏ định kiến, hận
thù, mặc cảm, hướng tới tương lai. Đại đoàn kết dân tộc, nhân dân luôn luôn gắn liền với
phát huy và hoàn thiện dân chủ xã hội chủ nghĩa, phát huy quyền làm chủ của nhân dân,
tôn trọng quyền con người, quyền và trách nhiệm công dân. Trong lịch sử, Đảng đã chú
trọng xây dụng các tổ chức Mặt trận, các đoàn thể để đoàn kết toàn dân, ngày nay, tăng
cường xây dựng Mặt trận Tổ quố
```

### MLN131 - PDF page 47

**PP-OCRv6**

```text
3. Phân tích đi tưng nghiên cúu ca ch nghīa xã
hi khoa hc? So sánh vi đi tưng nghiên cúu cúa
trit hoc?
4. Phân tích nhng đóng góp v lý lun chính tr - xã hi
ca Đng Cng sn Vit Nam qua 35 năm đi mi?
50
```

**Current corpus (Tesseract path)**

```text
3. Phân tích đối tượng nghiên cứu của chủ nghĩa xã
hội khoa học? So sánh với đối tượng nghiên cứu của
triết học?
4. Phân tích những đóng góp về lý luận chính trị - xã hội
của Đảng Cộng sản Việt Nam qua 35 năm đổi mới?
50
```

### MLN131 - PDF page 161

**PP-OCRv6**

```text
3. Bn cht và đnh hưng xây dng ch đ dân ch xā
hi ch nghīa  Vit Nam?
4. Ni dung và đnh hưng xây dng Nhà nưc pháp
quyn xã hi ch nghĩa  Vit Nam?
5. Liên h trách nhim cá nhân trong vic góp phn
xây dng nn dân ch xã hi ch nghĩa, Nhà nưc pháp
quyn xã hi ch nghĩa  nưc ta hin nay?
164
```

**Current corpus (Tesseract path)**

```text
3. Bản chất và định hướng xây dựng chế độ dân chủ xã
hội chủ nghĩa ở Việt Nam?
4. Nội dung và định hướng xây dựng Nhà nước pháp
quyền xã hội chủ nghĩa ở Việt Nam?
5. Liên hệ trách nhiệm cá nhân trong việc góp phần
xây dựng nền dân chủ xã hội chủ nghĩa, Nhà nước pháp
quyền xã hội chủ nghĩa ở nước ta hiện nay?
164
```

### MLN131 - PDF page 235

**PP-OCRv6**

```text
vn đ tôn giáo trong thi k quá đ lên ch nghĩa xã hi,
xây dng và bo v T quc xã hi ch nghĩa?
5. Phân tích mi quan h gia dân tc vi tôn giáo
Vit Nam và nh hưng ca mi quan h đó đn s n
đnh chính tr - xã hi ca đt nưc, đn đc lp, chù
quyn ca T quc? Trách nhim cá nhân?
238
```

**Current corpus (Tesseract path)**

```text
vấn đề tôn giáo trong thời kỳ quá độ lên chủ nghĩa xã hội,
xây dựng và bảo vệ Tổ quốc xã hội chủ nghĩa?
5. Phân tích mối quan hệ giữa dân tộc với tôn giáo ở
Việt Nam và ảnh hưởng của mối quan hệ đó đến sự ổn
định chính trị - xã hội của đất nước, đến độc lập, chủ
quyền của Tổ quốc? Trách nhiệm cá nhân?
288
```

### HCM202 - PDF page 72

**PP-OCRv6**

```text
t ngàn xưa đn nay gn lin vi truyên thng yêu nưc,
đu tranh chng gic ngoi xâm. Điu đó nói lên mt khát
khao to ln ca dân tc ta là luôn mong mun có đưc mt
nên đc lp cho dân tc, t do cho nhân dân và đó cūng là
mt giá tr tinh thân thiêng liêng, bât hú cúa dân tc mà
Hô Chí Minh là hin thân cho tinh thn y. Ngưi nói
rng: Cái mà tôi cn nht trên đi là đồng bào tôi đưc t
do, T quc tôi đưc đc lp¹.
Năm 1919, nhân dp các nưc Đồng minh thng trn
trong Chin tranh th gii thú nht hop Hi nghi
Vécxây (Pháp), thay mt nhūng ngưi Vit Nam yêu
nưc, Hồ Chí Minh đã gi ti Hi nghi bn Yêu sách ca
nhân dân An Nam, bao gm 8 đim vi hai ni dung
chính là đòi quyên bình đng v mt pháp lý và đòi các
quyên t do, dân chú ca ngưi dân Đông Dương. Bǎn
yêu sách không đưc Hi ngh chp nhn nhưng qua s
kin trên cho thy ln đu tiên, tư tưng Hồ Chí Minh
vê quyên ca các dân tc thuc đa mà trưc ht là
quyên bình đng và t do đā hình thành. Cǎn cú vào
nhng quyên t do, bình đng và quyên con ngưi -
"nhng quyên mà không ai có th xâm phm đưc” đā
đưc ghi trong bn Tuyên ngôn đc lp cúa cách mng
Mý nǎm 1776, Tuyên ngôn nhân quyên và dân quyên
ca Cách mng Pháp năm 1791, Hồ Chí Minh tip tc
khng đnh nhng giá tr thiêng liêng, bt bin v quyên
dân tc: "Tât că các dân tc trên th gii đu sinh ra
1. Xem Hồ Chí Minh: Toàn tp, Sđd, t.5, tr.201.
74
```

**Current corpus (Tesseract path)**

```text
vá
| từ ngàn xưa đến nay gắn liền với truyền thống yêu nước, : \
Ể đấu tranh chống giặc ngoại xâm. Điều đó nói lên một khát |
l_ khao to lớn của dân tộc ta là luôn mong muốn có được một h
. nền độc lập cho dân tộc, tự do cho nhân dân và đó cũng là ũ
một giá trị tính thần thiêng liêng, bất hủ của dân tộc mà ị
Hồ Chí Minh là hiện thân cho tỉnh thần ấy. Người nói ¡
rằng: Cái mà tôi cần nhất trên đời là đồng bào tôi được tự :
do, Tổ quốc tôi được độc lập". Ệ
Năm 1919, nhân địp các nước Đồng minh thắng trận <<
trong Chiến tranh thế giới thứ nhất họp Hội nghị :
Vécxây (Pháp), thay mặt những người Việt Nam yêu |
nước, Hồ Chí Minh đã gửi tới Hội nghị bần Yêu sách của
nhân dân An Nam, bao gồm 8 điểm với hai nội dung
chính là đòi quyền bình đẳng về mặt pháp lý và đòi các
quyền tự do, dân chủ của người dân Đông Dương. Bản
yêu sách không được Hội nghị chấp nhận nhưng qua sự Í
kiện trên cho thấy lần đầu tiên, tư tưởng Hồ Chí Minh .
về quyền của các dân tộc thuộc địa mà trước hết là l‹i
quyền bình đẳng và tự do đã hình thành. Căn cứ vào ị
những quyền tự do, bình đẳng và quyền con người - Ỉ
“những quyền mà không ai có thể xâm phạm được” đã ị
được ghi trong bản Tuyên ngôn độc lập của cách mạng
Mỹ năm 1776, Tuyên ngôn nhân quyền và dân quyền ị
của Cách mạng Pháp năm 1791, Hồ Chí Minh tiếp tục ị
khẳng định những giá trị thiêng liêng, bất biến về quyền |
dân tộc: “Tất cả các dân tộc trên thế giới đều sinh ra ị
1. Xem Hồ Chí Minh: Toàn tập, Sđủ, 5, tr.201. ¡
74 |
!
H
ị
LÍ
```

### HCM202 - PDF page 84

**PP-OCRv6**

```text
cách mng; liên lc vi tiu tư sn, trí thc, trung nông.
đ lôi kéo ho v phía vô sn giai cp; còn đi vi phú nông,
trung, tiu đia ch và tư sn Vit Nam mà chưa rõ mt
phăn cách mng thì phi li dng, ít ra cũng làm cho h
trung lp1.
Khi thc dân Pháp tin hành xâm lưc Vit Nam ln
thư hai, Hồ Chí Minh thit tha kêu gi mi ngưi không
phân bit giai tâng, dân tc, tôn giáo, đng phái... đoàn
kt đu tranh chng kě thù chung cúa dân tc. Trong
Li kêu goi toàn quc kháng chin (tháng 12/1946),
Ngưi vit: "Bát ky đàn ông, đàn bà, bát ky ngưi già,
ngưi trě, không chia tôn giáo, đng phái, dân tc. H là
ngưi Vit Nam thì phi đng lên đánh thc dân Pháp đ
cúu T quc”.
Trong khi xác đnh lc lưng cách mng là toàn dân,
Hồ Chí Minh lưu y rng, không đưc quên “công nông là
ngưòi chǔ cách mnh... là gc cách mnh". Trong tác
phm Đưng cách mnh, Ngưi giǎi thích: giai cp công
nhân và nông dân là hai giai cp đông đo và cách mng
nht, bi bóc lt nng nê nht, vì th "lòng cách mnh
càng bn, chí cách mnh càng quyt... công nông là tay
không chân ri, nu thua thì chi mt mt cái kip kh,
nu đưc thì đưc cå th gii, cho nên h gan góc”.
1. Xem Hồ Chí Minh: Toàn tp, Sđd, t.3, tr.3.
2. Hồ Chí Minh: Toàn tp, Sđd, t.4, tr.534.
3, 4. Hô Chí Minh: Toàn tp, Sđd, t.2, tr.288.
86
```

**Current corpus (Tesseract path)**

```text
TYị
cách mạng; liên lạc với tiểu tư sản, trí thức, trung nông... ( :
để lôi kéo họ về phía vô sản giai cấp; còn đối với phú nông,
trung, tiểu địa chủ và tư sắn Việt Nam mà chưa rõ mặt
phản cách mạng thì phải lợi dụng, ít ra cũng làm cho họ |
trung lập'. :
Khi thực dân Pháp tiến hành xâm lược Việt Nam lần
thứ hai, Hồ Chí Minh thiết tha kêu gọi mọi người không Ì
phân biệt giai tầng, dân tộc, tôn giáo, đẳng phái... đoàn .
kết đấu tranh chống kẻ thù chung của dân tộc. Trong í ñ
i Lời kêu gọi toàn quốc kháng chiến (tháng 12/1946), Ệ
Người viết: “Bất kỳ đàn ông, đàn bà, bất kỳ người già, Ệ
người trẻ, không chia tôn giáo, đẳng phái, dân tộc. Hễ là :
người Việt Nam thì phải đứng lên đánh thực dân Pháp để [
cứu Tổ quốc??, ị
Trong khi xác định lực lượng cách mạng là toàn dân,
Hồ Chí Minh lưu ý rằng, không được quên “công nông là
người chủ cách mệnh... là gốc cách mệnh”. Trong tác ¡ ¡
phẩm Đường cách mệnh, Người giải thích: giai cấp công ¡
| nhân và nông dân là hai giai cấp đông đảo và cách mạng ĩ
nhất, bị bóc lột nặng nề nhất, vì thế “lòng cách mệnh
| càng bền, chí cách mệnh càng quyết... công nông là tay
không chân rồi, nếu thua thì chỉ mất một cái kiếp khổ,
nếu được thì được cả thế giới, cho nên họ gan góc”. Ị
| |
1. Xem Hồ Chí Minh: Toàn tập, Sđd, t.8, tr3. ¡ |
2. Hồ Chí Minh: Toàn tập, Sdd, t.4, tr.534. ị
3, 4. Hồ Chí Minh: Toàn tập, Sđd, t.2, tr.288. ¡
H
```

### HCM202 - PDF page 267

**PP-OCRv6**

```text
MC LUC
Trang
Li Nhà xut bn
7
Chương 1
KHÁI NIĘM, DÕI TUQNG, PHUONG PHÁP
NGHIÊN CÚU VÀ Ý NGHÃA HQC TÂP
MÔN TU TUNG HÔ CHÍ MINH
11
A. Mc tiêu
11
B. Ni dung
12
I- Khái nim tư tưng Hồ Chí Minh
12
II- Đi tưng nghiên cu
19
III- Phưong pháp nghiên cu
20
IV- Y nghīa cúa vic hc tp môn hc Tư tưng
Hồ Chí Minh
28
C. Câu hi ôn tp
31
Chương 2
CO SÔ, QUÁ TRINH HINH THÀNH VÀ
PHÁT TRIĚN TU TUÔNG HÔ CHÍ MINH
32
A. Mc tiêu
32
B. Ni dung
33
I- Co s hình thành tư tưng Hồ Chí Minh
33
II- Quá trình hình thành và phát trin tư tưng
Hồ Chí Minh
50
269
```

**Current corpus (Tesseract path)**

```text
.
{
MỤC LỤC
‡
¡ Trang ¡
| Lời Nhà xuất bản 7 |
¡ „ Chương 1 |
¡ KHÁI NIỆM, ĐỐI TƯỢNG, PHƯƠNG PHÁP
¡ NGHIÊN CỨU VÀ Ý NGHĨA HỌC TẬP
Ï MÔN TƯ TƯỞNG HỒ CHÍ MINH 11 |
¡ A. Mục tiêu 11 |
¡ B. Nội dung 12 |
Ễ 1- Khái niệm tư tưởng Hồ Chí Minh 12
: 1I- Đối tượng nghiên cứu 19 ]
; TII- Phương pháp nghiên cứu 20 \
: TV- Ý nghĩa của việc học tập môn học Tư tưởng (
ị Hồ Chí Minh %8 :
C. Câu hỏi ôn tập 31 ¡
Chương 2
i GƠ SỞ, QUÁ TRÌNH HÌNH THÀNH VÀ
: PHÁT TRIỂN TƯ TƯỞNG HỒ CHÍ MINH 32
A. Mục tiêu 3% |
‡ B. Nội dung 33
: I- Gơ số hình thành tư tưởng Hồ Chí Minh 38
| II- Quá trình hình thành và phát triển tư tưởng
. Hồ Chí Minh 50 |
¡ 269 .
```
