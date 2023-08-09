# Multilingual Knowledge Retrieval (mKR)

# Installation
```
git clone https://github.com/panuthept/multilingual-knowledge-retrieval.git
cd multilingual-knowledge-retrieval
conda create -n mkr python
conda activate mkr
pip install -e .
conda install -c conda-forge faiss-cpu
```

# Running demo (mUSE Dense Retrieval)
```
python demo_dense_retrieval.py \
--query_file ./data/demo_queries.jsonl \
--doc_file ./data/demo_docs.jsonl \
--qrel_file ./data/demo_qrels.jsonl \
--index_file ./indexes/mUSE \
--top_k 3 \
--model_name mUSE
```

# Running demo (BM25 Sparse Retrieval)
```
python demo_sparse_retrieval.py \
--query_file ./data/demo_queries.jsonl \
--doc_file ./data/demo_docs.jsonl \
--qrel_file ./data/demo_qrels.jsonl \
--index_file ./indexes/bm25_okapi_newmm \
--top_k 3 \
--model_name bm25_okapi \
--tokenizer_name newmm
```

# Results
NOTE: [✅] means the document is relevant, [❌] means the document is not relevant.

Disclaimer: This demo only has 14 documents, so it is very likely that the relevant documents will be in the top 3.

# Dense Retrieval (mUSE) Results
## Query: ใครเป็นคนแต่งเรื่อง Harry Potter
### [✅] Score: 0.5733      Document[5]: แฮร์รี่ พอตเตอร์ (อังกฤษ: Harry Potter) เป็นชุดนวนิยายแฟนตาซีจำนวนเจ็ดเล่ม ประพันธ์โดยนักเขียนชาวอังกฤษชื่อว่า เจ. เค. โรว์ลิง เป็นเรื่องราวการผจญภัยของพ่อมดวัยรุ่น แฮร์รี่ พอตเตอร์ กับเพื่อนสองคน รอน วีสลีย์ และ เฮอร์ไมโอนี่ เกรนเจอร์ ซึ่งทั้งหมดเป็นนักเรียนของโรงเรียนคาถาพ่อมดแม่มดและเวทมนตร์ศาสตร์ฮอกวอตส์ โครงเรื่องหลักเกี่ยวกับภารกิจของแฮร์รี่ในการเอาชนะพ่อมดศาสตร์มืดที่ชั่วร้าย ลอร์ดโวลเดอมอร์ ผู้ที่ต้องการจะมีชีวิตเป็นอมตะ มีเป้าหมายเพื่อพิชิตมักเกิ้ล หรือประชากรที่ไม่มีอำนาจวิเศษ พิชิตโลกพ่อมดและทำลายทุกคนที่ขัดขวาง โดยเฉพาะอย่างยิ่ง แฮร์รี่ พอตเตอร์[1]

### [❌] Score: 0.4071      Document[7]: โรงเรียนคาถาพ่อมดแม่มดและเวทมนตร์ศาสตร์ฮอกวอตส์ (อังกฤษ: Hogwarts School of Witchcraft and Wizardry) ย่อเป็น ฮอกวอตส์ เป็นโรงเรียนสอนเวทมนตร์สมมติของประเทศสกอตแลนด์ ซึ่งเปิดสอนนักเรียนอายุระหว่างสิบเอ็ดถึงสิบแปดปี และเป็นฉากท้องเรื่องหลักในชุด แฮร์รี่ พอตเตอร์ และเป็นฉากหลักในโลกเวทมนตร์[3]

### [✅] Score: 0.3975      Document[6]: โจแอนน์ "โจ" โรว์ลิง (อังกฤษ: Joanne "Jo" Rowling, OBE FRSL[2]) หรือนามปากกา เจ. เค. โรว์ลิง[3] และโรเบิร์ต กัลเบรธ (เกิด 31 กรกฎาคม ค.ศ. 1965)[1] เป็นนักเขียนนวนิยายชาวอังกฤษ ผู้เป็นที่รู้จักกันดีในฐานะผู้ประพันธ์วรรณกรรมแฟนตาซีชุด แฮร์รี่ พอตเตอร์ ซึ่งได้รับความความสนใจจากทั่วโลก ได้รับรางวัลมากมาย และมียอดขายกว่า 500 ล้านเล่ม[4] และยังเป็นหนังสือชุดที่ขายดีที่สุดในประวัติศาสตร์[5] ด้านภาพยนตร์ชุดแฮร์รี่ พอตเตอร์ที่ดัดแปลงมาจากหนังสือก็เป็นภาพยนตร์ชุดที่ทำรายได้มากที่สุดเป็นอันดับสองในประวัติศาสตร์[6] โรว์ลิงอนุมัติบทภาพยนตร์ทุกภาค[7] และตลอดจนควบคุมงานฝ่ายสร้างสรรค์ภาพยนตร์ภาคสุดท้ายในฐานะผู้อำนวยการสร้าง[8]
------------------------------------------------------------------------------------------------------------------------------------------------------
## Query: ระเบิดปรมาณูถูกคิดค้นโดยใคร
### Top 3 documents
### [❌] Score: 0.1357      Document[2]: บิตคอยน์ (อังกฤษ: Bitcoin) เป็นคริปโทเคอร์เรนซี[10]:3 บิตคอยน์เป็นสกุลเงินดิจิทัลแรกที่ใช้ระบบกระจายอำนาจ โดยไม่มีธนาคารกลางหรือแม้แต่ผู้คุมระบบแม้แต่คนเดียว[10]:1[11] เครือข่ายเป็นแบบเพียร์ทูเพียร์ และการซื้อขายเกิดขึ้นระหว่างจุดต่อเครือข่าย (network node)โดยตรง ผ่านการใช้วิทยาการเข้ารหัสลับและไม่มีสื่อกลาง[10]:4 การซื้อขายเหล่านี้ถูกตรวจสอบโดยรายการเดินบัญชีแบบสาธารณะที่เรียกว่าบล็อกเชน บิตคอยน์ถูกพัฒนาโดยคนหรือกลุ่มคนภายใต้นามแฝง "ซาโตชิ นากาโมโตะ"[12] และถูกเผยแพร่ในรูปแบบซอฟต์แวร์โอเพนซอร์ซในปี พ.ศ. 2552[13]

### [✅] Score: 0.1298      Document[4]: Julius Robert Oppenheimer[note 1] (/ˈɒpənhaɪmər/ OP-ən-hy-mər; April 22, 1904 – February 18, 1967) was an American theoretical physicist and director of the Manhattan Project's Los Alamos Laboratory during World War II. He is often called the "father of the atomic bomb". Born in New York City to Jewish immigrants from Germany, Oppenheimer earned a bachelor's degree in chemistry from Harvard University in 1925 and a doctorate in physics from the University of Göttingen in Germany in 1927. After research at other institutions, he joined the physics department at the University of California, Berkeley, where he became a full professor in 1936. He made significant contributions to theoretical physics, including achievements in quantum mechanics and nuclear physics such as the Born–Oppenheimer approximation for molecular wave functions, work on the theory of electrons and positrons, the Oppenheimer–Phillips process in nuclear fusion, and the first prediction of quantum tunneling. With his students, he also made contributions to the theory of neutron stars and black holes, quantum field theory, and the interactions of cosmic rays.

### [❌] Score: 0.0977      Document[10]: ทาเกโอะ โอตสึกะ (ญี่ปุ่น: 大塚 剛央おおつか たけお; โรมาจิ: Ōtsuka Takeo, 19 ตุลาคม 1992[1] -) เป็น นักพากย์ชายชาวญี่ปุ่น สังกัดไอแอมเอนเทอร์ไพรส์[2] เกิดกิโตเกียว[2]
------------------------------------------------------------------------------------------------------------------------------------------------------
## Query: Who is an inventor of bitcoin?
### Top 3 documents
### [✅] Score: 0.3293      Document[2]: บิตคอยน์ (อังกฤษ: Bitcoin) เป็นคริปโทเคอร์เรนซี[10]:3 บิตคอยน์เป็นสกุลเงินดิจิทัลแรกที่ใช้ระบบกระจายอำนาจ โดยไม่มีธนาคารกลางหรือแม้แต่ผู้คุมระบบแม้แต่คนเดียว[10]:1[11] เครือข่ายเป็นแบบเพียร์ทูเพียร์ และการซื้อขายเกิดขึ้นระหว่างจุดต่อเครือข่าย (network node)โดยตรง ผ่านการใช้วิทยาการเข้ารหัสลับและไม่มีสื่อกลาง[10]:4 การซื้อขายเหล่านี้ถูกตรวจสอบโดยรายการเดินบัญชีแบบสาธารณะที่เรียกว่าบล็อกเชน บิตคอยน์ถูกพัฒนาโดยคนหรือกลุ่มคนภายใต้นามแฝง "ซาโตชิ นากาโมโตะ"[12] และถูกเผยแพร่ในรูปแบบซอฟต์แวร์โอเพนซอร์ซในปี พ.ศ. 2552[13]

### [❌] Score: 0.1422      Document[4]: Julius Robert Oppenheimer[note 1] (/ˈɒpənhaɪmər/ OP-ən-hy-mər; April 22, 1904 – February 18, 1967) was an American theoretical physicist and director of the Manhattan Project's Los Alamos Laboratory during World War II. He is often called the "father of the atomic bomb". Born in New York City to Jewish immigrants from Germany, Oppenheimer earned a bachelor's degree in chemistry from Harvard University in 1925 and a doctorate in physics from the University of Göttingen in Germany in 1927. After research at other institutions, he joined the physics department at the University of California, Berkeley, where he became a full professor in 1936. He made significant contributions to theoretical physics, including achievements in quantum mechanics and nuclear physics such as the Born–Oppenheimer approximation for molecular wave functions, work on the theory of electrons and positrons, the Oppenheimer–Phillips process in nuclear fusion, and the first prediction of quantum tunneling. With his students, he also made contributions to the theory of neutron stars and black holes, quantum field theory, and the interactions of cosmic rays.

### [❌] Score: 0.1219      Document[12]: ขอให้โชคดีมีชัยในโลกแฟนตาซี! (ญี่ปุ่น: この素晴らしい世界に祝福を; โรมาจิ: Kono Subarashii Sekai ni Shukufuku wo!) เป็นนวนิยายญี่ปุ่นที่เขียนโดย Natsume Akatsuki เนื้อเรื่องเกี่ยวกับเด็กผู้ชายคนหนึ่งที่ถูกส่งไปยังโลกแฟนตาซีหลังการตายของเขา และเขาได้สร้างปาร์ตี้กับเทพธิดา สาวน้อยนักเวทและทหารครูเสดเพื่อต่อสู้กับมอนส์เตอร์ในโลกแฟนตาซี โดยเริ่มต้นจากซีรีส์นวนิยายทางเว็บที่เผยแพร่ใน Shousetsuka ni Narou ระหว่างเดือนธันวาคม 2012 และตุลาคม 2013 ซีรีส์ดังกล่าวได้รับการแก้ไขเป็นซีรีส์นวนิยายแสงพิมพ์ที่มีภาพประกอบโดย Kurone Mishima ซึ่งเริ่มตีพิมพ์ภายใต้ สำนักพิมพ์คาโดกาวะ
------------------------------------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------------------------------------

# Sparse Retrieval (BM25) Results
## Query: ใครเป็นคนแต่งเรื่อง Harry Potter
### [✅] Score: 8.6386      Document[5]: แฮร์รี่ พอตเตอร์ (อังกฤษ: Harry Potter) เป็นชุดนวนิยายแฟนตาซีจำนวนเจ็ดเล่ม ประพันธ์โดยนักเขียนชาวอังกฤษชื่อว่า เจ. เค. โรว์ลิง เป็นเรื่องราวการผจญภัยของพ่อมดวัยรุ่น แฮร์รี่ พอตเตอร์ กับเพื่อนสองคน รอน วีสลีย์ และ เฮอร์ไมโอนี่ เกรนเจอร์ ซึ่งทั้งหมดเป็นนักเรียนของโรงเรียนคาถาพ่อมดแม่มดและเวทมนตร์ศาสตร์ฮอกวอตส์ โครงเรื่องหลักเกี่ยวกับภารกิจของแฮร์รี่ในการเอาชนะพ่อมดศาสตร์มืดที่ชั่วร้าย ลอร์ดโวลเดอมอร์ ผู้ที่ต้องการจะมีชีวิตเป็นอมตะ มีเป้าหมายเพื่อพิชิตมักเกิ้ล หรือประชากรที่ไม่มีอำนาจวิเศษ พิชิตโลกพ่อมดและทำลายทุกคนที่ขัดขวาง โดยเฉพาะอย่างยิ่ง แฮร์รี่ พอตเตอร์[1]

### [❌] Score: 4.4653      Document[0]: พลเอก ประยุทธ์ จันทร์โอชา (เกิด 21 มีนาคม พ.ศ. 2497) ชื่อเล่น ตู่ เป็นนักการเมืองและทหารบกชาวไทย นายกรัฐมนตรีไทยคนที่ 29 และรัฐมนตรีว่าการกระทรวงกลาโหมคนปัจจุบัน เคยดำรงตำแหน่งผู้บัญชาการทหารบก ตั้งแต่ พ.ศ. 2553 จนถึง พ.ศ. 2557 และหัวหน้าคณะรักษาความสงบแห่งชาติ ซึ่งก่อรัฐประหารใน พ.ศ. 2557 และเป็นคณะรัฐประหารที่ปกครองประเทศไทยตั้งแต่ 22 พฤษภาคม พ.ศ. 2557 จนถึง 16 กรกฎาคม พ.ศ. 2562 เป็นระยะ 5 ปี 1 เดือน 3 สัปดาห์ 3 วัน

### [❌] Score: 4.0275      Document[2]: บิตคอยน์ (อังกฤษ: Bitcoin) เป็นคริปโทเคอร์เรนซี[10]:3 บิตคอยน์เป็นสกุลเงินดิจิทัลแรกที่ใช้ระบบกระจายอำนาจ โดยไม่มีธนาคารกลางหรือแม้แต่ผู้คุมระบบแม้แต่คนเดียว[10]:1[11] เครือข่ายเป็นแบบเพียร์ทูเพียร์ และการซื้อขายเกิดขึ้นระหว่างจุดต่อเครือข่าย (network node)โดยตรง ผ่านการใช้วิทยาการเข้ารหัสลับและไม่มีสื่อกลาง[10]:4 การซื้อขายเหล่านี้ถูกตรวจสอบโดยรายการเดินบัญชีแบบสาธารณะที่เรียกว่าบล็อกเชน บิตคอยน์ถูกพัฒนาโดยคนหรือกลุ่มคนภายใต้นามแฝง "ซาโตชิ นากาโมโตะ"[12] และถูกเผยแพร่ในรูปแบบซอฟต์แวร์โอเพนซอร์ซในปี พ.ศ. 2552[13]
------------------------------------------------------------------------------------------------------------------------------------------------------
## Query: ระเบิดปรมาณูถูกคิดค้นโดยใคร
### Top 3 documents
### [❌] Score: 4.1865      Document[2]: บิตคอยน์ (อังกฤษ: Bitcoin) เป็นคริปโทเคอร์เรนซี[10]:3 บิตคอยน์เป็นสกุลเงินดิจิทัลแรกที่ใช้ระบบกระจายอำนาจ โดยไม่มีธนาคารกลางหรือแม้แต่ผู้คุมระบบแม้แต่คนเดียว[10]:1[11] เครือข่ายเป็นแบบเพียร์ทูเพียร์ และการซื้อขายเกิดขึ้นระหว่างจุดต่อเครือข่าย (network node)โดยตรง ผ่านการใช้วิทยาการเข้ารหัสลับและไม่มีสื่อกลาง[10]:4 การซื้อขายเหล่านี้ถูกตรวจสอบโดยรายการเดินบัญชีแบบสาธารณะที่เรียกว่าบล็อกเชน บิตคอยน์ถูกพัฒนาโดยคนหรือกลุ่มคนภายใต้นามแฝง "ซาโตชิ นากาโมโตะ"[12] และถูกเผยแพร่ในรูปแบบซอฟต์แวร์โอเพนซอร์ซในปี พ.ศ. 2552[13]

### [❌] Score: 3.0887      Document[12]: ขอให้โชคดีมีชัยในโลกแฟนตาซี! (ญี่ปุ่น: この素晴らしい世界に祝福を; โรมาจิ: Kono Subarashii Sekai ni Shukufuku wo!) เป็นนวนิยายญี่ปุ่นที่เขียนโดย Natsume Akatsuki เนื้อเรื่องเกี่ยวกับเด็กผู้ชายคนหนึ่งที่ถูกส่งไปยังโลกแฟนตาซีหลังการตายของเขา และเขาได้สร้างปาร์ตี้กับเทพธิดา สาวน้อยนักเวทและทหารครูเสดเพื่อต่อสู้กับมอนส์เตอร์ในโลกแฟนตาซี โดยเริ่มต้นจากซีรีส์นวนิยายทางเว็บที่เผยแพร่ใน Shousetsuka ni Narou ระหว่างเดือนธันวาคม 2012 และตุลาคม 2013 ซีรีส์ดังกล่าวได้รับการแก้ไขเป็นซีรีส์นวนิยายแสงพิมพ์ที่มีภาพประกอบโดย Kurone Mishima ซึ่งเริ่มตีพิมพ์ภายใต้ สำนักพิมพ์คาโดกาวะ

### [❌] Score: 0.0977      Document[10]: ทาเกโอะ โอตสึกะ (ญี่ปุ่น: 大塚 剛央おおつか たけお; โรมาจิ: Ōtsuka Takeo, 19 ตุลาคม 1992[1] -) เป็น นักพากย์ชายชาวญี่ปุ่น สังกัดไอแอมเอนเทอร์ไพรส์[2] เกิดกิโตเกียว[2]
------------------------------------------------------------------------------------------------------------------------------------------------------
## Query: Who is an inventor of bitcoin?
### Top 3 documents
### [❌] Score: 10.4001     Document[4]: Julius Robert Oppenheimer[note 1] (/ˈɒpənhaɪmər/ OP-ən-hy-mər; April 22, 1904 – February 18, 1967) was an American theoretical physicist and director of the Manhattan Project's Los Alamos Laboratory during World War II. He is often called the "father of the atomic bomb". Born in New York City to Jewish immigrants from Germany, Oppenheimer earned a bachelor's degree in chemistry from Harvard University in 1925 and a doctorate in physics from the University of Göttingen in Germany in 1927. After research at other institutions, he joined the physics department at the University of California, Berkeley, where he became a full professor in 1936. He made significant contributions to theoretical physics, including achievements in quantum mechanics and nuclear physics such as the Born–Oppenheimer approximation for molecular wave functions, work on the theory of electrons and positrons, the Oppenheimer–Phillips process in nuclear fusion, and the first prediction of quantum tunneling. With his students, he also made contributions to the theory of neutron stars and black holes, quantum field theory, and the interactions of cosmic rays.

### [❌] Score: 9.7008      Document[13]: Prayut Chan-o-cha (sometimes spelled Prayuth Chan-ocha; Thai: ประยุทธ์ จันทร์โอชา, pronounced [prā.jút tɕān.ʔōː.tɕʰāː]; born 21 March 1954) is a Thai politician and army officer[1] who has served as the Prime Minister of Thailand since he seized power in a military coup in 2014. He is concurrently the Minister of Defence, a position he has held in his own government since 2019.[2] Prayut served as Commander-in-Chief of the Royal Thai Army from 2010 to 2014[3][4] and led the 2014 Thai coup d'état which installed the National Council for Peace and Order (NCPO), the military junta which governed Thailand between 22 May 2014 and 10 July 2019.[5]

### [❌] Score: 7.3714      Document[7]: โรงเรียนคาถาพ่อมดแม่มดและเวทมนตร์ศาสตร์ฮอกวอตส์ (อังกฤษ: Hogwarts School of Witchcraft and Wizardry) ย่อเป็น ฮอกวอตส์ เป็นโรงเรียนสอนเวทมนตร์สมมติของประเทศสกอตแลนด์ ซึ่งเปิดสอนนักเรียนอายุระหว่างสิบเอ็ดถึงสิบแปดปี และเป็นฉากท้องเรื่องหลักในชุด แฮร์รี่ พอตเตอร์ และเป็นฉากหลักในโลกเวทมนตร์[3]
------------------------------------------------------------------------------------------------------------------------------------------------------