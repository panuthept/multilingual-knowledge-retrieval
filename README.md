# multilingual-knowledge-retrieval

# Installation
```
git clone https://github.com/panuthept/multilingual-knowledge-retrieval.git
cd multilingual-knowledge-retrieval
conda create -n mkr python
conda activate mkr
pip install -e .
```

# Running demo
```
python demo.py --query_file ./data/demo_queries.jsonl --doc_file ./data/demo_docs.jsonl --qrel_file ./data/demo_qrels.jsonl --index_file ./data/demo_doc_embs.npy --top_k 3
```

# Results
NOTE: [✅] means the document is relevant, [❌] means the document is not relevant.

Disclaimer: This demo only has 14 documents, so it is very likely that the relevant documents will be in the top 3.
## Query: ลุงตู่ (ประยุทธ์) ทำอาชีพอะไร
### Top 3 documents
### [❌] Score: 0.1654      Document[1]: นักการเมือง, ผู้นำทางการเมืองหรือบุคคลทางการเมือง (อังกฤษ: politician มาจากภาษากรีกโบราณ polis ที่แปลว่า "เมือง") เป็นบุคคลผู้เกี่ยวข้องกับการมีอิทธิพลต่อนโยบายสาธารณะและการวินิจฉัยสั่งการ ซึ่งรวมผู้ดำรงตำแหน่งวินิจฉัยสั่งการในรัฐบาล และผู้ที่มุ่งดำรงตำแหน่งเหล่านั้น ไม่ว่าด้วยวิธีการเลือกตั้ง การสืบทอด รัฐประหาร การแต่งตั้ง การพิชิต หรือวิธีอื่น การเมืองไม่ใช่จำกัดอยู่เพียงวิธีการปกครองผ่านตำแหน่งสาธารณะเท่านั้น ตำแหน่งทางการเมืองอาจถืออยู่ใน

### [✅] Score: 0.1488      Document[13]: Prayut Chan-o-cha (sometimes spelled Prayuth Chan-ocha; Thai: ประยุทธ์ จันทร์โอชา, pronounced [prā.jút tɕān.ʔōː.tɕʰāː]; born 21 March 1954) is a Thai politician and army officer[1] who has served as the Prime Minister of Thailand since he seized power in a military coup in 2014. He is concurrently the Minister of Defence, a position he has held in his own government since 2019.[2] Prayut served as Commander-in-Chief of the Royal Thai Army from 2010 to 2014[3][4] and led the 2014 Thai coup d'état which installed the National Council for Peace and Order (NCPO), the military junta which governed Thailand between 22 May 2014 and 10 July 2019.[5]

### [❌] Score: 0.1129      Document[4]: Julius Robert Oppenheimer[note 1] (/ˈɒpənhaɪmər/ OP-ən-hy-mər; April 22, 1904 – February 18, 1967) was an American theoretical physicist and director of the Manhattan Project's Los Alamos Laboratory during World War II. He is often called the "father of the atomic bomb". Born in New York City to Jewish immigrants from Germany, Oppenheimer earned a bachelor's degree in chemistry from Harvard University in 1925 and a doctorate in physics from the University of Göttingen in Germany in 1927. After research at other institutions, he joined the physics department at the University of California, Berkeley, where he became a full professor in 1936. He made significant contributions to theoretical physics, including achievements in quantum mechanics and nuclear physics such as the Born–Oppenheimer approximation for molecular wave functions, work on the theory of electrons and positrons, the Oppenheimer–Phillips process in nuclear fusion, and the first prediction of quantum tunneling. With his students, he also made contributions to the theory of neutron stars and black holes, quantum field theory, and the interactions of cosmic rays.
------------------------------------------------------------------------------------------------------------------------------------------------------
## Query: Who is an inventor of bitcoin?
### Top 3 documents
### [✅] Score: 0.3534      Document[2]: บิตคอยน์ (อังกฤษ: Bitcoin) เป็นคริปโทเคอร์เรนซี[10]:3 บิตคอยน์เป็นสกุลเงินดิจิทัลแรกที่ใช้ระบบกระจายอำนาจ โดยไม่มีธนาคารกลางหรือแม้แต่ผู้คุมระบบแม้แต่คนเดียว[10]:1[11] เครือรืายเป็นแบบเทูยทูทูเทูยทู และการซื้อขายเกิดกินระหจุางจุดจุอเครือรืาย (network node)โดยตรง ผ่านการใช้วิทยาการเหัารหัสหับและไม่มีม่อกลาง[10]:4 การซื้อขายเหถูาถูถูกตรวจสอบโดยรายการเดินบัญดิแบบสาธารณะที่เล็ยกว่าบล็อกเชน ถูตคอยน์ถูกพัฒนาโดยคนหพัอกลุ่มคนภายใต้นามแฝง "ซาโตชิ นากาโมโตะ"[12] และถูกเผยแพร่ในถูปแบบซอฟต์แวร์โอเพนซอร์ซในร์ พ.ศ. 2552[13]

### [❌] Score: 0.1337      Document[3]: ธนาคารกลาง (อังกฤษ: central bank, reserve bank หรือ monetary authority) เป็นหน่วยงานที่จัดตั้งขึ้นเพื่อใช้เป็นหน่วยงานกลางในการดำเนินการทางด้านการเงินของประเทศ ในประเทศไทยหน่วยงานที่ทำหน้าน้ธนาคารกลางคือ ธนาคารแห่งประเทศไทย

### [❌] Score: 0.0677      Document[12]: ขอให้โชคดีมีชัยในโลกแฟนตาซี! (ญี่ปุ่น: この素晴らしい世界に祝福を; โรมาจิ: Kono Subarashii Sekai ni Shukufuku wo!) เป็นนวนิยายญี่ปุ่นที่เขียนโดย Natsume Akatsuki เรื่อเรื่องเรื่ยวด็บเด็กด็ชายคนหถูงถูถูกถูงไปยังโลกแฟนตาซีหซีงการตายของเขา และเขาได้สด้างปาร้ตี้ร้บเทพกัดา สาวนัอยนักเวทและทหารครูเสดเรูอรูอรูรูบมอนรูเตอร์ในโลกแฟนตาซี โดยเริ่มต้นจากซีรีส์นวรียายทางเว็บร่เผยแพร่ใน Shousetsuka ni Narou ระหว่างเว่อนธันวาคม 2012 และตุลาคม 2013 ส์รีส์ดังกส์าวไส์ดับการแซีไขเซีนซีรีซีนวรียายแสงส์มพ์นิส์ภาพประกอบโดย Kurone Mishima ซึ่งเริ่มตีพิมพ์ภายใต้ สำนักต้มพ์คาโดกาวะ
------------------------------------------------------------------------------------------------------------------------------------------------------
## Query: ธนาคารกลางทำหน้าที่อะไร
### Top 3 documents
### [✅] Score: 0.4993      Document[3]: ธนาคารกลาง (อังกฤษ: central bank, reserve bank หรือ monetary authority) เป็นหน่วยงานที่จัดตั้งขึ้นเพื่อใช้เป็นหน่วยงานกลางในการดำเนินการทางด้านการเงินของประเทศ ในประเทศไทยหน่วยงานที่ทำหน้าน้ธนาคารกลางคือ ธนาคารแห่งประเทศไทย

### [❌] Score: 0.1422      Document[2]: บิตคอยน์ (อังกฤษ: Bitcoin) เป็นคริปโทเคอร์เรนซี[10]:3 บิตคอยน์เป็นสกุลเงินดิจิทัลแรกที่ใช้ระบบกระจายอำนาจ โดยไม่มีธนาคารกลางหรือแม้แต่ผู้คุมระบบแม้แต่คนเดียว[10]:1[11] เครือรืายเป็นแบบเทูยทูทูเทูยทู และการซื้อขายเกิดกินระหจุางจุดจุอเครือรืาย (network node)โดยตรง ผ่านการใช้วิทยาการเหัารหัสหับและไม่มีม่อกลาง[10]:4 การซื้อขายเหถูาถูถูกตรวจสอบโดยรายการเดินบัญดิแบบสาธารณะที่เล็ยกว่าบล็อกเชน ถูตคอยน์ถูกพัฒนาโดยคนหพัอกลุ่มคนภายใต้นามแฝง "ซาโตชิ นากาโมโตะ"[12] และถูกเผยแพร่ในถูปแบบซอฟต์แวร์โอเพนซอร์ซในร์ พ.ศ. 2552[13]

### [❌] Score: 0.0813      Document[1]: นักการเมือง, ผู้นำทางการเมืองหรือบุคคลทางการเมือง (อังกฤษ: politician มาจากภาษากรีกโบราณ polis ที่แปลว่า "เมือง") เป็นบุคคลผู้เกี่ยวข้องกับการมีอิทธิพลต่อนโยบายสาธารณะและการนินิจนิยนิงการ ซึ่งรวมซึ่ดำรงตำแหนิงนินิจนิยนิงการในรัฐบาล และผู้ที่ผู้งดำรงตำแหน่งเหล่าล่น ไล่นั้านั้วยนั้ธีการเตั้อกตั้ง การสืบทอด รัฐประหาร การแต่งต่ง การพิชิต หชิอชิธีชิน การเมืองไม่ใยู่จำกัดอม่เพียงวิพีการปกครองธีานตำแหน่งสาธารณะเน่านั้น ตำแหน่งทางการเถืองอาจถืออยู่ใน
------------------------------------------------------------------------------------------------------------------------------------------------------
## Query: ระเบิดปรมาณูถูกคิดค้นโดยใคร
### Top 3 documents
### [❌] Score: 0.1357      Document[2]: บิตคอยน์ (อังกฤษ: Bitcoin) เป็นคริปโทเคอร์เรนซี[10]:3 บิตคอยน์เป็นสกุลเงินดิจิทัลแรกที่ใช้ระบบกระจายอำนาจ โดยไม่มีธนาคารกลางหรือแม้แต่ผู้คุมระบบแม้แต่คนเดียว[10]:1[11] เครือข่ายเป็นแบบเพียร์ทูเพียร์ และการซื้อขายเกิดขึ้นระหว่างจุดต่อเครือข่าย (network node)โดยตรง ผ่านการใช้วิทยาการเข้ารหัสลับและไม่มีสื่อกลาง[10]:4 การซื้อขายเหล่านี้ถูกตรวจสอบโดยรายการเดินบัญชีแบบสาธารณะที่เรียกว่าบล็อกเชน บิตคอยน์ถูกพัฒนาโดยคนหรือกลุ่มคนภายใต้นามแฝง "ซาโตชิ นากาโมโตะ"[12] และถูกเผยแพร่ในรูปแบบซอฟต์แวร์โอเพนซอร์ซในปี พ.ศ. 2552[13]

### [✅] Score: 0.1298      Document[4]: Julius Robert Oppenheimer[note 1] (/ˈɒpənhaɪmər/ OP-ən-hy-mər; April 22, 1904 – February 18, 1967) was an American theoretical physicist and director of the Manhattan Project's Los Alamos Laboratory during World War II. He is often called the "father of the atomic bomb". Born in New York City to Jewish immigrants from Germany, Oppenheimer earned a bachelor's degree in chemistry from Harvard University in 1925 and a doctorate in physics from the University of Göttingen in Germany in 1927. After research at other institutions, he joined the physics department at the University of California, Berkeley, where he became a full professor in 1936. He made significant contributions to theoretical physics, including achievements in quantum mechanics and nuclear physics such as the Born–Oppenheimer approximation for molecular wave functions, work on the theory of electrons and positrons, the Oppenheimer–Phillips process in nuclear fusion, and the first prediction of quantum tunneling. With his students, he also made contributions to the theory of neutron stars and black holes, quantum field theory, and the interactions of cosmic rays.

### [❌] Score: 0.0977      Document[10]: ทาเกโอะ โอตสึกะ (ญี่ปุ่น: 大塚 剛央おおつか たけお; โรมาจิ: Ōtsuka Takeo, 19 ตุลาคม 1992[1] -) เป็น นักพากย์ชายชาวญี่ปุ่น สังกัดไอแอมเอนเทอร์ไพรส์[2] เกิดกิโตเกียว[2]
------------------------------------------------------------------------------------------------------------------------------------------------------
## Query: ใครเป็นคนแต่งเรื่อง Harry Potter
### Top 3 documents
### [✅] Score: 0.5733      Document[5]: แฮร์รี่ พอตเตอร์ (อังกฤษ: Harry Potter) เป็นชุดนวนิยายแฟนตาซีจำนวนเจ็ดเล่ม ประพันธ์โดยนักเขียนชาวอังกฤษชื่อว่า เจ. เค. โรว์ลิง เป็นเรื่องราวการผจญภัยของพ่อมดวัยวัน แฮรุ่รี่ พอตเตอร์ ร์บเร์อนสองคน รอน ย์สย์ย์ และ เฮอร์ไมโอร์ เกรนเจอร์ ร์งร์งหมดเป็นป็กเป็ยนของโรงเรียนคาถาพ่อมดแพ่มดและเวทมนตร์ศาสตร์ฮอกวอตส์ โครงเรื่องหรื่กเกัยวกับภารกัจของแฮร์รี่ในการเอาชนะพ่อมดศาสตพ่มืดที่พ่วร์าย ลอร์ดโวลเดอมอร์ ผู้ที่ร์องการจะผู้ที่ต้ตเที่นอมตะ ต้เป้าหมายเพื่อกิ้ป้ตมักเกิ้ล หกิ้อประชากรที่ไม่วิอำนาจชิเศษ พ่ชิตโลกพ่อมดและทำลายทุกคนทุขัดขวาง โดยเฉพาะอย่างรี่ง แฮร์รี่ พอตเตอร์[1]

### [❌] Score: 0.4071      Document[7]: โรงเรียนคาถาพ่อมดแม่มดและเวทมนตร์ศาสตร์ฮอกวอตส์ (อังกฤษ: Hogwarts School of Witchcraft and Wizardry) ย่อเป็น ฮอกวอตส์ เป็นโรงเรียนสอนเวทมนตร์สมมติของประเทศสกอตแลนด์ ด์งเด์ดสอนรีกเรียนอารีระหรีางอ็บเอ็ดอ็งอ็บแปดสิ และเป็นฉากป็องเรื่องหลักในชุด แฮชุรี่ พอตเตอร์ และเป็นฉากหลักในโลกเวทมนตร์[3]

### [✅] Score: 0.3975      Document[6]: โจแอนน์ "โจ" โรว์ลิง (อังกฤษ: Joanne "Jo" Rowling, OBE FRSL[2]) หรือนามปากกา เจ. เค. โรว์ลิง[3] และโรเบิร์ต กัลเบรธ (เกิด 31 กรกฎาคม ค.ศ. 1965)[1] เป็นป็กเป็ยนนวป็ยายชาวอังกฤษ ผู้เผู้นผู้ผู้ผู้กผู้นผู้ในฐานะผู้ประพันพัวรรณกรรมแฟนตาซีชุด แฮชุรี่ พอตเตอร์ ร์งไร์รับความความสนใจจากทั่วโลก ได้รับรางรัลมากมาย และมียอดขายกว่า 500 ล้านเล่ม[4] และนังเป็นหสืงสือสืดสืขายสืที่สืดในประที่นัศาสตร์[5] นัานภาพยนตรี่นัดแฮนัรี่ พอตเตอร์ร์รี่ดแปลงมาจากหรี่งสือรี่เสืนภาพยนตก็ชุดที่ทำรายไก็มากก็สุดเก็นก็นก็บสองในประร์ติศาสตร์[6] โรติลิงอติลิติบทภาพยนตมัติกภาค[7] และตลอดจนควบร้มงานฝ่ายสร้างสรรร้ภาพยนตร์ภาคสุดร์ายในฐานะร์อำนวยการสร้าง[8]
------------------------------------------------------------------------------------------------------------------------------------------------------
## Query: ใครเป็นคนพากย์เสียงเมกุมินในเรื่อง Konosuba
### Top 3 documents
### [✅] Score: 0.4361      Document[9]: ริเอะ ทากาฮาชิ (ญี่ปุ่น: 高橋 李依; โรมาจิ: Takahashi Rie; เกิด 27 กุมภาพันธ์ ค.ศ. 1994) เป็นนักพากย์เสียงและนักร้องชาวญี่ปุ่น และยังเป็นหนึ่งในสมาชิกตัวแทนของ 81 พรอดิวดิ[1] หลังจากลัไลัเลัาลัวมเป็นป็กพากนัเนัยง เธอได้เด้นเด้นด้ตาบะ อิชิโนเสะจากอนิเมะเนิอง "โซเระกะเซยู!" และเธอป็งเป็นสมาป็กในวง Earphones อีกอีวย คุณทากาฮาด้ ได้พากย์เสียงตัวละครในอนิเมะอรื่หลายเรื่อง เช่น Konosuba รับบทเป็นเมป็รัน Re:Zero รับบทเลีนเอลีเลีย Karakai Jozu no Takagi-san รับบทเกินทาคากิและ Fate/Grand Order รับบทเป็นแมช ไครีไลท์ Witchy Precure! รับบทเป็นมิไร อาซาฮิน/Cure Miracle Bakugan: Battle Planet รับบทเป็นดัน คูโซ และวีดีเกม Genshin Impact รับบทเป็น Hu Tao นอกจากนี้แล้ว ยัณทากาฮาชิยังไล้มีโอกาสมีองเพลงประกอบอนิเมะอยู่หลายเรื่อง จนไลิรื่บรางวัลชนะเน้ศวักแสดงหยี่งหน้าใหม่ยอดเยี่ยมในงาน 10th Seiyu Awards

### [❌] Score: 0.3352      Document[10]: ทาเกโอะ โอตสึกะ (ญี่ปุ่น: 大塚 剛央おおつか たけお; โรมาจิ: Ōtsuka Takeo, 19 ตุลาคม 1992[1] -) เป็น นักพากย์ชายชาวญี่ปุ่น สังกัดไอแอมเอนเทอร์ไพรส์[2] เกิดกิโตเกียว[2]

### [❌] Score: 0.3220      Document[11]: ยูมิ อูจิยามะ (ญี่ปุ่น: 内山 夕実; โรมาจิ: Uchiyama Yumi; 30 ตุลาคม ค.ศ. 1987[1] -) เป็นนักพากย์หญิง ชาวญี่ปุ่น เกิดที่โตเกียว[1] สังกัดสำนักงานโอซาวะ[2]