# Nghiên cứu chuyên sâu về LeWorldModel và cách biến nó thành một demo World Model khả thi trong Unity

## Tóm tắt điều tra nhanh

`leworldmodel` **không phải là một từ viết sai mơ hồ nữa**. Tính đến tháng 5 năm 2026, nó là tên chính thức của một paper và codebase nghiên cứu mới: **LeWorldModel** hay viết tắt là **LeWM**. Paper gốc có tiêu đề *LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels* trên arXiv; repo chính thức là `lucas-maes/le-wm`; website chính thức là `le-wm.github.io`. Đây là một **world model dạng JEPA** học trực tiếp từ pixel, dùng **2 loss terms**, có khoảng **15M tham số**, có thể train trên **một GPU trong vài giờ**, và theo tác giả thì cho tốc độ lập kế hoạch nhanh hơn tới **48×** so với các world model dựa trên foundation model trong cùng setup thí nghiệm. citeturn30search0turn31view0turn41search1turn42view3turn42view4

Điểm rất quan trọng cho mục tiêu của bạn: **LeWM hiện là code research-oriented**, dựa trên `stable-worldmodel` để quản lý môi trường/planning/evaluation, `stable-pretraining` để train, dữ liệu ở dạng **HDF5 offline trajectories**, và các benchmark chính được công bố là **TwoRoom, Reacher, PushT, OGBench-Cube**. Trong các nguồn chính thức tôi kiểm tra, **không thấy tài liệu tích hợp chính thức với Unity hoặc Unity ML-Agents**. Vì vậy, nếu mục tiêu là làm demo Unity thuyết trình, cách đúng và an toàn nhất là: **dùng LeWorldModel như nền tảng khoa học để giải thích**, còn phần demo nên được triển khai thành **“ứng dụng ý tưởng World Model vào AI agent trong Unity”**, thay vì hứa rằng bạn đang dùng nguyên bản repo LeWM trong game Unity. citeturn29view1turn29view2turn41search1turn42view0turn42view1turn42view2

Nói thẳng theo góc nhìn researcher + engineer: **LeWM là nguồn tham khảo học thuật rất tốt**, nhưng **không phải một framework game-ready**. Nó hợp để bạn dùng làm “trục lý thuyết + paper core”, sau đó xây một demo Unity theo tinh thần world model đơn giản hơn, dễ chạy hơn, và dễ bảo vệ hơn trước giảng viên. Đây là chiến lược có xác suất thành công cao nhất. citeturn31view0turn29view1turn32view0turn16view0

**LeWorldModel là gì / có thể là gì**

| Loại nguồn | Tên nguồn | Link | Nội dung chính | Độ liên quan | Độ tin cậy | Ghi chú |
|---|---|---|---|---|---|---|
| Paper chính thức | LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels | arXiv citeturn30search0turn41search5 | Paper gốc, mô tả ý tưởng JEPA world model, SIGReg, benchmark, kết quả | Rất cao | Rất cao | Đây là định danh chuẩn của `leworldmodel` |
| Repo chính thức | `lucas-maes/le-wm` | GitHub citeturn31view0turn29view1 | Code chính thức, lệnh cài đặt, train/eval, checkpoint, dependency | Rất cao | Rất cao | MIT license; research code, không phải Unity plugin |
| Website chính thức | `le-wm.github.io` | Project page citeturn41search1 | Tóm tắt approach, planning từ pixel, ví dụ trực quan | Rất cao | Rất cao | Dễ dùng để đưa vào slide |
| Checkpoint và dữ liệu | Hugging Face models/datasets được repo trỏ tới | HF collection qua repo citeturn29view1turn29view2 | Checkpoint cho PushT, Cube, TwoRooms, Reacher; dataset offline | Cao | Cao | Hữu ích nếu bạn muốn chạy lại benchmark gốc |
| Phụ thuộc chính | `stable-pretraining` | GitHub citeturn7view1turn31view0 | Framework PyTorch/Lightning dùng cho pretraining/world models | Trung bình | Cao | Cho thấy LeWM tựa lên hạ tầng research riêng |
| Nguồn liên quan gần nhất | DreamerV3 | Paper + repo citeturn10search1turn33search0turn32view0 | Hướng world model mạnh, phổ biến, tổng quát hơn trong RL | Cao | Rất cao | Phù hợp làm đối chiếu nếu LeWM không hợp demo Unity |

**Cách nói an toàn trong báo cáo/thuyết trình**

Bạn nên nói theo câu này:

> “Sau khi khảo sát nguồn chính thức, `LeWorldModel` là một world model học từ pixel theo hướng JEPA, công bố năm 2026. Tuy nhiên, bản phát hành hiện tại là research code cho benchmark điều khiển dựa trên dữ liệu offline, chưa có pipeline Unity chính thức. Vì vậy, phần thực nghiệm của em chuyển thành: **ứng dụng ý tưởng World Model lấy cảm hứng từ LeWorldModel vào AI agent trong game Unity**.” citeturn30search0turn31view0turn41search1

Cách nói này vừa đúng sự thật, vừa tránh bị bắt lỗi “nói quá” về mức độ tích hợp thực tế.

## Giải thích World Model bằng Feynman

**World Model là gì?**  
World Model là một mô hình mà AI học ra để **dự đoán môi trường sẽ thay đổi thế nào khi nó làm một hành động**. Nó không chỉ phản ứng kiểu “thấy gì làm nấy”, mà cố gắng xây một “mô hình thế giới bên trong đầu” để tưởng tượng tương lai trước khi hành động. Ý tưởng này đã được thể hiện rất rõ trong các dòng công trình như *World Models*, PlaNet, Dreamer, MuZero và sau này là DreamerV3. citeturn10search2turn10search0turn11search0turn12search7turn33search0

Ví dụ trong game 2D, agent đang đứng trước một con quái, một hố sâu, một item và một cánh cửa. Nếu agent có world model, nó có thể “thử trong đầu” các phương án như đi thẳng, né sang trái, nhảy, lùi lại, hoặc đánh từ phía sau; sau đó chọn hành động nào có khả năng dẫn đến trạng thái tốt hơn. World model vì thế đặc biệt hữu ích khi môi trường phức tạp, reward thưa, hoặc hành động hiện tại ảnh hưởng mạnh đến kết quả tương lai. PlaNet và Dreamer nổi tiếng chính ở chỗ cho thấy learned latent dynamics có thể giúp điều khiển từ pixel hiệu quả hơn về dữ liệu so với nhiều hướng model-free thuần túy. citeturn10search0turn11search0turn10search1

```text
Quan sát hiện tại + hành động dự định
        ↓
World Model dự đoán trạng thái tiếp theo
        ↓
AI so sánh nhiều tương lai khả dĩ
        ↓
Chọn hành động có kết quả tốt nhất
```

**Nói đơn giản là...**  
World Model giống như **trí tưởng tượng của AI**. Trước khi làm thật, nó thử “mơ thử” xem nếu mình làm A, B, C thì điều gì sẽ xảy ra. Công trình *World Models* của Ha và Schmidhuber thậm chí còn nổi tiếng vì ý tưởng agent có thể học và luyện trong một “giấc mơ” do world model sinh ra. citeturn10search2turn9search0

**Vì sao AI cần World Model?**  
Vì nếu mỗi lần muốn biết kết quả của một hành động, AI đều phải thử ngoài đời thật hoặc chạy thật trong môi trường, quá trình học sẽ rất tốn sample, chậm và đôi khi nguy hiểm. Model-based RL sinh ra để giải quyết chuyện đó: học mô hình động lực môi trường rồi dùng mô hình đó cho planning hoặc cho imagined rollouts. PlaNet dùng latent-space planning; Dreamer học actor-critic trên imagined trajectories; MuZero học model chỉ của những đại lượng quan trọng cho planning như reward, policy, value; DreamerV3 đẩy cách làm này tới hơn 150 tác vụ với một cấu hình tương đối thống nhất. citeturn10search0turn11search0turn12search7turn33search0

**World Model khác gì với các cách khác?**  
Reinforcement Learning model-free học trực tiếp “nên làm gì” nhưng không học rõ “thế giới vận hành ra sao”. Behavior Tree và Finite State Machine thì dựa vào logic do con người viết sẵn. Imitation Learning học bắt chước dữ liệu chuyên gia. LLM agent mạnh về ngôn ngữ và planning ký hiệu, nhưng không mặc định có mô hình động lực vật lý của game. World Model nằm ở chỗ khác: nó học **cơ chế chuyển trạng thái** của môi trường, rồi dùng nó để ra quyết định. MuZero và Dreamer là hai ví dụ rất rõ: cả hai đều không chỉ học policy, mà còn học một internal model để planning hoặc imagination. citeturn12search7turn11search0turn10search1

**Thử hình dung bằng game 2D**

- Nhân vật thấy quái trước mặt.
- World model dự đoán: nếu lao thẳng vào, máu sẽ giảm.
- Nếu đi vòng sau lưng quái, xác suất an toàn cao hơn.
- Nếu bước thêm 2 tile bên phải, có thể rơi hố.
- Nếu nhảy đúng lúc, có thể lấy item rồi tới cửa.

AI không cần phải được lập trình cứng “đi vòng sau lưng quái”. Nó có thể **học ra** rằng tương lai đó tốt hơn.

**Nói đơn giản là...**  
Rule-based AI giống như học thuộc đáp án. World Model giống như **hiểu bài**.

## Lịch sử phát triển và phân tích kỹ thuật

Dòng world model hiện đại xuất hiện từ nhu cầu vượt qua nhược điểm của **model-free RL**, vốn thường cần rất nhiều tương tác môi trường. *World Models* năm 2018 cho thấy có thể học biểu diễn nén và động lực của môi trường rồi dùng chúng cho control; PlaNet năm 2019 cho thấy có thể planning trực tiếp trong latent space từ pixel; Dreamer năm 2019–2020 mở rộng sang học hành vi bằng latent imagination; DreamerV2 chạm mức human-level trên Atari bằng discrete world model; MuZero học model “value-equivalent” để tree search mà không cần biết luật game; DreamerV3 cho thấy world model có thể tổng quát hơn qua hơn 150 nhiệm vụ; và LeWorldModel năm 2026 tập trung vào một câu hỏi khác: **làm sao train JEPA world model từ pixel một cách ổn định, đơn giản, ít hack hơn**. citeturn10search2turn10search0turn11search0turn11search1turn12search7turn33search0turn30search0

| Năm | Công trình | Ý tưởng chính | Tác động |
|---|---|---|---|
| 2018 | World Models | Học mô hình generative của môi trường và dùng “dream” để train policy nhỏ gọn citeturn10search2turn9search0 | Mở lại sự chú ý mạnh vào learned world models từ pixel |
| 2019 | PlaNet | Planning trong latent space bằng learned dynamics model có thành phần stochastic + deterministic citeturn10search0turn10search4 | Chứng minh planning-from-pixels khả thi hơn |
| 2019–2020 | Dreamer | Học actor-critic từ imagined latent rollouts citeturn11search0turn8search1 | Làm world model thực dụng hơn cho control dài hạn |
| 2020–2021 | DreamerV2 | Discrete world models, human-level Atari citeturn11search1turn32view1 | World model lần đầu thật sự cạnh tranh mạnh ở Atari |
| 2019–2020 | MuZero | Học model phục vụ planning: reward, value, policy, transition; không cần biết luật citeturn12search7turn12search0 | Kết hợp model learning + tree search rất mạnh |
| 2023–2025 | DreamerV3 | Một cấu hình Dreamer dùng trên hơn 150 tasks, kể cả Minecraft diamond citeturn10search1turn33search0turn32view0 | Đưa world model tiến gần “general RL recipe” |
| 2026 | LeWorldModel | JEPA world model train end-to-end từ pixel chỉ với prediction loss + Gaussian latent regularizer | Nhấn mạnh **sự ổn định và đơn giản hóa training** citeturn30search0turn41search1turn41search5 |

**Thành phần kỹ thuật thường có trong World Model**

```text
Game Screen / Observation
        ↓
Encoder
        ↓
Latent State
        ↓
Dynamics Model ──→ Future Latent States
        ↓
Reward Prediction
        ↓
Policy chọn hành động tốt nhất
```

- **Encoder**: nhận observation thô, thường là hình ảnh, rồi nén thành latent state. Trong LeWM, encoder biến frame quan sát thành embedding gọn hơn để predictor làm việc. citeturn41search1turn41search5  
- **Latent State**: biểu diễn “tình trạng quan trọng” của môi trường ở dạng nén. Nó không cần giữ mọi pixel; nó chỉ cần giữ cái gì hữu ích cho dự đoán và điều khiển. Những công trình như PlaNet, Dreamer và LeWM đều xoay mạnh quanh latent state. citeturn10search0turn11search0turn41search1  
- **Dynamics Model**: dự đoán latent tiếp theo khi biết latent hiện tại và hành động. Đây là “động cơ tưởng tượng tương lai”. LeWM website mô tả rất rõ predictor dự đoán embedding của frame tiếp theo từ embedding hiện tại và action. citeturn41search1turn41search5  
- **Reward Model**: đoán phần thưởng để planning/RL biết nhánh nào có lợi. Dreamer và MuZero dùng reward prediction rõ ràng. LeWM gốc, trong release hiện tại, thiên về goal-conditioned latent planning hơn là actor-critic reward learning kiểu Dreamer. citeturn11search0turn12search7turn41search1  
- **Policy / Actor**: chọn hành động. Dreamer học actor trong imagination; MuZero dùng model + search; LeWM hiện dùng planning bằng CEM từ ảnh start/goal trong latent space thay vì public release của một actor RL kiểu Dreamer. citeturn11search0turn12search7turn41search1  
- **Value / Critic**: ước lượng hành động tốt xấu trong dài hạn. Rất quan trọng ở Dreamer/MuZero, nhưng **không phải thành phần trung tâm** của LeWM release như ở Dreamer. citeturn11search0turn12search7turn41search1  
- **Imagination Rollout**: mô phỏng tương lai trong latent space thay vì chạy môi trường thật. Đây là điểm chung của Dreamer và world-model family nói chung. citeturn11search0turn32view0

**Điểm kỹ thuật đáng nhớ nhất của LeWM**

LeWM dùng một encoder và một predictor để làm JEPA world model, tối ưu bởi **prediction loss** cộng với **SIGReg**, một regularizer ép latent phân bố kiểu Gaussian để tránh collapse. Tác giả nhấn mạnh đây là JEPA đầu tiên train ổn định end-to-end từ raw pixels với chỉ hai loss terms, giảm số hyperparameter loss nhạy cảm xuống còn một hệ số chính. Ở test time, LeWM encode ảnh đầu và ảnh mục tiêu vào latent space rồi dùng **Cross-Entropy Method** để tối ưu chuỗi action sao cho latent cuối gần goal latent nhất. citeturn41search1turn30search0turn42view3

**Nói đơn giản là...**  
Dreamer thường giống “học một bộ não gồm trí tưởng tượng + người ra quyết định”.  
MuZero giống “học phần thế giới đủ để search nước đi tốt”.  
LeWM giống “học một trí tưởng tượng nén từ pixel thật ổn định, rồi dùng nó để tìm chuỗi hành động dẫn đến ảnh đích”.

## So sánh với các phương pháp khác và ứng dụng thực tế

Bảng dưới đây kết hợp **dữ kiện từ literature world model** với **đánh giá triển khai thực tế trong game Unity** của tôi.

| Phương pháp | Ưu điểm | Nhược điểm | Khi nào dùng | Có phù hợp Unity game không? |
|---|---|---|---|---|
| Rule-based AI | Nhanh, dễ debug, deterministic | Cứng, khó mở rộng, không học | NPC đơn giản, puzzle, boss script | Rất phù hợp cho game nhỏ |
| Finite State Machine | Cấu trúc rõ, dễ dạy | Bùng nổ trạng thái khi game phức tạp | Enemy có vài mode như patrol/chase/attack | Phù hợp |
| Behavior Tree | Tốt hơn FSM cho logic phức tạp | Vẫn cần designer viết tay | NPC có hành vi có thứ tự ưu tiên | Phù hợp |
| Pathfinding A* | Rất mạnh cho đi đường | Không giải quyết combat/tactics/học | Navigation trên map/grid/navmesh | Rất phù hợp |
| RL model-free | Có thể học policy mạnh trực tiếp | Tốn sample, khó reward shaping, khó giải thích | Khi chỉ cần agent học chơi | Phù hợp, nhất là với ML-Agents citeturn16view0turn19search13 |
| Imitation Learning | Học nhanh nếu có expert demo | Bị giới hạn bởi chất lượng expert | Có sẵn dữ liệu người chơi tốt | Phù hợp vừa phải citeturn16view0 |
| LLM Agent | Giỏi ra quyết định ngôn ngữ mức cao | Không mặc định hiểu physics/game dynamics frame-level | Quest logic, narrative AI, planner cấp cao | Chỉ phù hợp tầng meta, không thay game control |
| World Model | Có thể tưởng tượng tương lai, sample-efficient hơn, planning tốt hơn trong dài hạn | Phức tạp hơn, dễ bias nếu model sai, khó tích hợp | Nhiệm vụ cần planning, sparse reward, dynamics khó | Phù hợp nếu demo nghiên cứu hoặc game AI nâng cao citeturn10search0turn11search0turn12search7turn33search0 |

**Tại sao người ta dùng World Model?**  
Vì nó cho AI khả năng **học luật vận hành của môi trường**, chứ không chỉ học phản xạ. PlaNet, Dreamer và DreamerV3 đều cho thấy world model có thể tận dụng tưởng tượng trong latent space để cải thiện hiệu quả học và planning; MuZero cho thấy nếu chỉ học đúng phần cần cho planning thì vẫn cực mạnh. citeturn10search0turn11search0turn12search7turn33search0

**Nó mạnh ở đâu?**  
Mạnh ở môi trường mà hành động hiện tại gây hệ quả dài hạn, reward thưa, hoặc mô phỏng thật tốn kém. Đó là lý do world models xuất hiện rất nhiều trong control, robotics, và game benchmark. DreamerV3 còn mở rộng sang một tập rất lớn miền ứng dụng; V-JEPA 2 cho thấy world-model style prediction/planning còn đi sang robotics zero-shot. citeturn33search0turn13search1turn13search5

**Nó yếu ở đâu?**  
Nó yếu khi mô hình học sai động lực thật. Lúc đó agent có thể “tưởng tượng sai nhưng rất tự tin”. Ngoài ra, pipeline world model thường dài: thu dữ liệu, train encoder/dynamics/reward/policy, kiểm chứng latent, kiểm chứng planning. Với game Unity nhỏ và mục tiêu là demo nhanh, phần engineering thường là vấn đề lớn hơn lý thuyết. Đây chính là lý do tôi khuyên bạn dùng hướng **world-model-inspired** thay vì cố nhồi full LeWM research stack vào Unity. citeturn31view0turn29view1turn32view0

**Khi nào không nên dùng?**  
Nếu bạn chỉ cần 3–5 hành vi rõ ràng cho NPC, hoặc deadline rất ngắn, FSM/BT/RL baseline thường hợp lý hơn. Với game Unity nhỏ, world model chỉ đáng dùng khi mục tiêu của bạn là **trình diễn ý tưởng AI hiện đại**, so sánh baseline với prediction-based agent, hoặc làm đề tài học thuật. citeturn16view0turn10search1

**Ứng dụng thực tế**

| Ứng dụng | World Model giúp gì? | Ví dụ cụ thể | Độ khó triển khai | Giá trị demo |
|---|---|---|---|---|
| Game AI tự chơi | Tưởng tượng trước kết quả hành động | Bot tự tìm đường qua bẫy, nhặt item, tới cửa | Trung bình | Rất cao |
| NPC thông minh | Học state transitions, tránh va chạm, chọn góc đánh | Kẻ địch đoán vị trí tương lai của người chơi | Trung bình | Cao |
| Procedural gameplay testing | Chạy agent để tìm lỗi map, soft-lock, reward exploit | ML-Agents đã nêu use case automated testing game builds | Trung bình | Cao citeturn16view0 |
| Robotics | Học dynamics, giảm thử-sai ngoài đời thật | Dreamer, V-JEPA 2-AC cho planning robot | Cao | Cao citeturn11search0turn13search1 |
| Autonomous driving / simulation | Dự đoán tình huống tương lai, risk-aware planning | World model cho driving scene prediction | Cao | Trung bình |
| LLM agent / planning | Bổ sung tầng “mô hình thế giới” thay vì thuần text | LeCun AMI và JEPA vision nhấn vào predictive world model | Cao | Trung bình citeturn14search0turn13search0 |
| Game tutorial / coach AI | Dự đoán người chơi sắp thất bại ở đâu | Gợi ý “đừng nhảy ở đây, dễ rơi hố” | Trung bình | Cao |

**Nói đơn giản là...**  
World Model đáng dùng khi bạn muốn AI **không chỉ biết làm**, mà còn **biết vì sao việc đó sẽ dẫn tới điều gì**.

## Ứng dụng vào Unity

Sau khi so sánh các repo Unity 2D/pixel open-source có license rõ ràng, tôi thấy bạn nên đánh giá theo hai trục: **độ dễ chạy** và **độ dễ biến thành environment cho agent**. Về mặt này, repo không cần phải là game thương mại hoàn chỉnh; quan trọng hơn là nó phải có gameplay rõ, scene rõ, luật rõ, và dễ gắn observation/action/reward. citeturn22view7turn26view3turn24view3turn21view5turn22view2turn40view2

| Repo | Loại game | Unity version | Mức hoàn chỉnh | Có dễ chạy không? | License | Vì sao phù hợp / không phù hợp với World Model | Link |
|---|---|---|---|---|---|---|---|
| Ryadel/2DRogueTest | Top-down roguelike pixel prototype | README: Unity 2022+; `ProjectVersion.txt`: 6000.0.37f1 | Prototype rõ ràng, playable | Dễ: README có `SampleScene.unity` và cách chạy | MIT | **Rất phù hợp**: top-down 2D, action space đơn giản, dễ tạo scene train riêng, khớp tốt với Unity 6 + ML-Agents | citeturn22view7turn25view2 |
| batuhancetinkaya1/2D-Platformer-Game | Platformer 2D có combat/boss | 2022.3.44f1 | Khá đầy đủ | Dễ: README có clone/open/play | MIT | Phù hợp nếu bạn muốn demo platformer, nhưng combat + boss + nhiều mechanic làm reward design khó hơn | citeturn26view3turn26view2 |
| YagmurCemGul/my-awesome-game | Platformer 2D nhặt diamond và né hazard | README: 2022.3 LTS+; raw file: 2023.2.3f1 | Nhỏ gọn, rõ luật | Tương đối dễ | MIT | **Rất phù hợp về gameplay**: collect-avoid-finish rất hợp RL/world-model demo; điểm trừ là version README và project file hơi lệch nhau | citeturn24view3turn25view0 |
| wjoh0315/REDBLUE-Unity2DGame | Platformer co-op | README: 2021.3.24f1; raw file: 2022.3.40f1 | Playable | Dễ | MIT | Dùng được, nhưng co-op hai nhân vật làm environment phức tạp hơn nhiều cho demo đầu tiên | citeturn21view5turn25view3 |
| kennedyvnak/unity-metroidvania | Metroidvania open-source | README + raw file: Unity 6000.0.32f1 | Tương đối lớn | Trung bình | MIT | Hấp dẫn về mặt trình bày, nhưng quá lớn cho demo agent đầu tiên | citeturn22view2turn27view3 |
| Walkator/Kailius | Platformer pixel-art | raw file: 2020.1.1f1 | Hoàn chỉnh hơn về game feel | Trung bình | MIT | Có thể dùng, nhưng version cũ hơn, mobile-oriented hơn, không tiện bằng Unity 6 stack hiện tại | citeturn40view2turn27view0 |

**Repo được khuyến nghị**  
Nếu ưu tiên của bạn là **tỷ lệ chạy được cao**, **ít xung đột tool**, **dễ gắn ML-Agents**, và **dễ giải thích trong buổi thuyết trình**, tôi khuyên chọn **`Ryadel/2DRogueTest`**. Lý do chính là repo này có scene chạy rõ ràng, code khá sạch, gameplay top-down đơn giản hơn platformer về mặt RL, license MIT rõ, và project file đang ở Unity 6 — hợp thẳng với line cài đặt ML-Agents hiện tại dùng Unity 6000.0+. citeturn22view7turn25view2turn16view1

Nếu giảng viên thích game nhìn “ra game” hơn là prototype top-down, lựa chọn số 2 của tôi là **`my-awesome-game`** hoặc **`2D-Platformer-Game`**. Hai repo này có gameplay trực quan hơn cho khán giả phổ thông, nhưng việc train agent trong platformer thường khó hơn đáng kể vì timing nhảy, hazard collision, và long-horizon credit assignment. citeturn24view3turn26view3

**Thiết kế demo cụ thể**

Tôi đề xuất demo theo hướng **an toàn nhưng vẫn đúng tinh thần world model**:

| Thành phần | Thiết kế đề xuất | Lý do |
|---|---|---|
| Game được chọn | `2DRogueTest`, nhưng tạo thêm một **TrainingRoom** dựa trên asset và code sẵn có | Không train trực tiếp trên dungeon procedural đầy đủ; dễ kiểm soát hơn |
| Mục tiêu agent | Di chuyển tới goal tile, né enemy, sống sót, nếu có thể thì tấn công khi cần | Khán giả nhìn vào là hiểu ngay |
| Observation space | Vector state: vị trí agent, velocity, HP, vector tới goal, vector tới enemy gần nhất, khoảng cách tường cục bộ, cờ “can attack” | Dễ train hơn visual obs; dễ debug |
| Action space | 2 continuous actions cho X/Y movement + 1 discrete action cho attack | Top-down 2D rất hợp kiểu này |
| Reward function | +0.01 nếu tiến gần goal, +1 khi chạm goal, -0.2 khi bị hit, -1 khi chết, +0.2 khi đẩy lùi/kết liễu enemy, -0.001 mỗi bước | Reward rõ ràng, giúp agent học dần |
| Training loop | Baseline PPO bằng ML-Agents trước; sau đó thêm world model một bước dự đoán next state/reward | Có baseline để so sánh |
| Model architecture | Baseline: PPO trong ML-Agents. World-model demo: MLP dynamics + reward head trên vector observations; planning ngắn hạn k bước | Dễ triển khai hơn full JEPA/Dreamer |
| Công cụ | Unity + ML-Agents + Python + TensorBoard | Stack chính thống, tài liệu tốt citeturn16view0turn19search13turn36view0 |
| Đánh giá agent | Episode return, success rate tới goal, time-to-goal, damage taken, number of enemy contacts | Có số để đưa lên slide |
| Cách trình bày demo | Chiếu 3 video: rule-based / PPO baseline / PPO + prediction module | Rất thuyết phục cho hội đồng |

**Kiến trúc triển khai World Model trong Unity**

| Hướng | Độ khó | Tính đúng với World Model | Khả năng demo thành công | Thời gian làm | Khuyến nghị |
|---|---|---:|---:|---:|---|
| A — Unity ML-Agents + model-free RL | Thấp | Thấp | Rất cao | 3–5 ngày | Bắt buộc làm baseline |
| B — ML-Agents + next-state predictor đơn giản | Trung bình | Trung bình | Cao | 1–2 tuần | **Khuyến nghị mạnh nhất cho thuyết trình** |
| C — World model riêng từ trajectories Unity | Cao | Cao | Trung bình | 2–4 tuần | Tốt nếu bạn có thêm thời gian |
| D — Dreamer/LeWM-style external pipeline nối với Unity | Rất cao | Rất cao | Thấp–trung bình | 3–6 tuần | Chỉ nên làm như hướng nghiên cứu nâng cao |

**Khuyến nghị thực tế theo thời gian**

- **Nếu chỉ có 1 tuần**: làm **A + một ít B**. Tức là PPO baseline chắc chắn chạy được, rồi thêm predictor một bước để có “mùi world model”.
- **Nếu có 2 tuần**: làm **A + B hoàn chỉnh**, có biểu đồ so sánh agent có/không có prediction.
- **Nếu có 1 tháng**: tiến lên **C**, thu trajectory từ Unity, train world model riêng, thử short-horizon latent planning hoặc model-predictive control đơn giản.

## Hướng dẫn cài đặt từng bước và code mẫu

Phần này tôi viết theo phương án khuyến nghị: **Unity 6 + 2DRogueTest + ML-Agents baseline + world-model extension đơn giản**.

**Bước chuẩn bị công cụ**

Git cho Windows có trang cài đặt chính thức riêng; bản mới nhất trên trang cài đặt Windows tại thời điểm tôi kiểm tra là **2.54.0 x64**. Unity khuyến nghị cài Editor qua **Unity Hub**, và archive editor chính thức cho phép cài các bản cũ/chính xác theo version. Với ML-Agents Python package, PyPI hiện ghi **`mlagents` 1.1.0** và yêu cầu **Python >=3.10.1, <=3.10.12**; tài liệu cài đặt của ML-Agents cũng khuyến nghị Python 3.10.12 và Unity 6000.0 trở lên. citeturn37search4turn39search1turn37search1turn36view0turn16view1

**Lệnh và thao tác nên làm**

```bash
# 1) Clone game mẫu
git clone https://github.com/Ryadel/2DRogueTest.git
cd 2DRogueTest

# 2) Clone ML-Agents toolkit để lấy package + sample configs
git clone --branch release_22 https://github.com/Unity-Technologies/ml-agents.git
```

Repo `2DRogueTest` có README hướng dẫn mở `Scenes/SampleScene.unity`; raw `ProjectVersion.txt` chỉ ra project đang ở **Unity 6000.0.37f1**, nên bạn nên cài đúng bản này trong Unity Hub để giảm rủi ro. README của ML-Agents installation hiện ghi workflow cài với Unity 6000.0+, Python 3.10.12 và có lệnh clone branch `release_22`; đồng thời PyPI của `mlagents` hiện là 1.1.0. Vì tài liệu và release page có một chút drift theo thời gian, cách an toàn là **pin đúng Python 3.10.12** và dùng **một release family nhất quán** giữa package C# và Python. citeturn22view7turn25view2turn16view1turn36view0turn17search1

**Cài Python env cho ML-Agents**

```powershell
py -3.10 -m venv .venv_mla
.\.venv_mla\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install mlagents==1.1.0 tensorboard
```

Cách kiểm tra thành công:

```powershell
python --version
mlagents-learn --help
python -c "import mlagents; print(mlagents.__version__)"
```

Nếu `python --version` không ra 3.10.x, bạn đang dùng sai interpreter. Đây là lỗi phổ biến nhất ở bước này. Yêu cầu version được nêu rõ trên PyPI và tài liệu cài đặt của ML-Agents. citeturn36view0turn16view1

**Cài package ML-Agents vào Unity project**

Trong Unity:

1. Mở `2DRogueTest` bằng Unity Hub với Editor **6000.0.37f1**.  
2. Vào `Window > Package Manager`.  
3. Theo docs chính thức, bạn có thể cài `com.unity.ml-agents` trực tiếp từ registry hoặc add local package từ repo ML-Agents đã clone bằng cách chọn `ml-agents/com.unity.ml-agents/package.json`. citeturn16view1

**Cách kiểm tra thành công trong Unity**

- Package Manager thấy `ML Agents`.
- Trong menu tạo component, tìm được `Behavior Parameters`, `Decision Requester`, `Agent`.
- Console không báo lỗi compile do package.

**Mở game và xác minh project chạy**

Repo `2DRogueTest` ghi rõ:

- Mở project bằng Unity 2022 trở lên.
- Mở `Scenes/SampleScene.unity`.
- Nhấn Play để chạy dungeon được sinh. citeturn22view7

Trong thực tế, do `ProjectVersion.txt` đang ở Unity 6, bạn nên mở bằng Unity 6. Khi scene chạy được, hãy chụp màn hình hoặc quay 10–15 giây làm bằng chứng “baseline game works”.

**Tạo scene training**

Tôi khuyên **không train trực tiếp trên dungeon procedural** ở vòng đầu. Hãy:

- Duplicate `SampleScene` thành `TrainingRoom`.
- Tắt procedural generator nếu có.
- Đặt agent spawn cố định.
- Đặt 1 goal object.
- Đặt 1 enemy đơn giản hoặc obstacle tĩnh.
- Gắn script `PlayerAgent.cs` vào player mới hoặc override player hiện có.

Điều này biến open-source game thành **một environment nhỏ, dễ học, dễ giải thích**.

**Code mẫu tối thiểu cho Unity ML-Agents**

_File gợi ý: `Assets/Scripts/ML/PlayerAgent.cs`_

```csharp
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using UnityEngine;

[RequireComponent(typeof(Rigidbody2D))]
public class PlayerAgent : Agent
{
    [Header("Scene References")]
    public Transform goal;
    public Transform enemy;
    public Transform spawnPoint;

    [Header("Movement")]
    public float moveSpeed = 4f;

    private Rigidbody2D rb;
    private Vector2 startPos;
    private float lastDistanceToGoal;

    public override void Initialize()
    {
        rb = GetComponent<Rigidbody2D>();
        if (spawnPoint != null)
        {
            startPos = spawnPoint.position;
        }
        else
        {
            startPos = transform.position;
        }
    }

    public override void OnEpisodeBegin()
    {
        rb.linearVelocity = Vector2.zero;
        rb.angularVelocity = 0f;

        transform.position = startPos;

        if (goal != null)
        {
            lastDistanceToGoal = Vector2.Distance(transform.position, goal.position);
        }
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        // Agent position
        sensor.AddObservation(transform.position.x);
        sensor.AddObservation(transform.position.y);

        // Agent velocity
        sensor.AddObservation(rb.linearVelocity.x);
        sensor.AddObservation(rb.linearVelocity.y);

        // Relative vector to goal
        if (goal != null)
        {
            Vector2 toGoal = goal.position - transform.position;
            sensor.AddObservation(toGoal.x);
            sensor.AddObservation(toGoal.y);
            sensor.AddObservation(toGoal.magnitude);
        }
        else
        {
            sensor.AddObservation(0f);
            sensor.AddObservation(0f);
            sensor.AddObservation(0f);
        }

        // Relative vector to nearest enemy
        if (enemy != null)
        {
            Vector2 toEnemy = enemy.position - transform.position;
            sensor.AddObservation(toEnemy.x);
            sensor.AddObservation(toEnemy.y);
            sensor.AddObservation(toEnemy.magnitude);
        }
        else
        {
            sensor.AddObservation(0f);
            sensor.AddObservation(0f);
            sensor.AddObservation(0f);
        }
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        float moveX = Mathf.Clamp(actions.ContinuousActions[0], -1f, 1f);
        float moveY = Mathf.Clamp(actions.ContinuousActions[1], -1f, 1f);

        Vector2 move = new Vector2(moveX, moveY);
        rb.linearVelocity = move * moveSpeed;

        // Small step penalty to encourage faster completion
        AddReward(-0.001f);

        if (goal != null)
        {
            float currentDistance = Vector2.Distance(transform.position, goal.position);

            // Reward progress toward goal
            float progress = lastDistanceToGoal - currentDistance;
            AddReward(progress * 0.02f);

            lastDistanceToGoal = currentDistance;
        }
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var continuous = actionsOut.ContinuousActions;
        continuous[0] = Input.GetAxisRaw("Horizontal");
        continuous[1] = Input.GetAxisRaw("Vertical");
    }

    private void OnTriggerEnter2D(Collider2D other)
    {
        if (other.CompareTag("Goal"))
        {
            AddReward(1.0f);
            EndEpisode();
        }
        else if (other.CompareTag("Enemy") || other.CompareTag("Hazard"))
        {
            AddReward(-1.0f);
            EndEpisode();
        }
    }
}
```

Đây là skeleton tối thiểu, bám đúng vòng đời Agent của ML-Agents: `CollectObservations`, `OnActionReceived`, `Heuristic`, `OnEpisodeBegin`. API này là cách dùng chuẩn trong docs ML-Agents. citeturn16view0turn19search13

**YAML training config mẫu**

_File gợi ý: `config/rogue_ppo.yaml`_

```yaml
behaviors:
  RogueAgent:
    trainer_type: ppo
    hyperparameters:
      batch_size: 1024
      buffer_size: 10240
      learning_rate: 3.0e-4
      beta: 5.0e-3
      epsilon: 0.2
      lambd: 0.95
      num_epoch: 3
      learning_rate_schedule: linear
    network_settings:
      normalize: true
      hidden_units: 128
      num_layers: 2
    reward_signals:
      extrinsic:
        gamma: 0.99
        strength: 1.0
    max_steps: 300000
    time_horizon: 128
    summary_freq: 10000
```

**Lệnh train**

```powershell
.\.venv_mla\Scripts\Activate.ps1
mlagents-learn config/rogue_ppo.yaml --run-id=rogue_baseline --time-scale=20
```

Sau đó quay lại Unity và bấm Play. `mlagents-learn` là entry point chính thức cho workflow training của ML-Agents. citeturn15search16turn19search13turn36view0

**Xem TensorBoard**

```powershell
tensorboard --logdir results
```

Mục tiêu tối thiểu là thấy:

- cumulative reward tăng dần,
- episode length giảm hoặc ổn định hợp lý,
- success rate tăng theo thời gian nếu bạn log thêm metric custom.

**Đưa model vào Unity để inference**

ML-Agents dùng model inference trong Unity thông qua **Sentis**; tài liệu cũng nêu sample workflow chạy pre-trained `.onnx` model ngay trong Unity. Nghĩa là sau khi train xong, bạn gắn model ONNX vào `Behavior Parameters` để agent chạy inference trong editor hoặc build. citeturn19search1turn19search10turn19search6

**Bản world model đơn giản bằng Python**

_File gợi ý: `wm_train.py`_

```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class TrajectoryDataset(Dataset):
    def __init__(self, data):
        self.obs = torch.tensor(data["obs"], dtype=torch.float32)
        self.act = torch.tensor(data["act"], dtype=torch.float32)
        self.next_obs = torch.tensor(data["next_obs"], dtype=torch.float32)
        self.rew = torch.tensor(data["rew"], dtype=torch.float32).unsqueeze(-1)

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        return self.obs[idx], self.act[idx], self.next_obs[idx], self.rew[idx]

class DynamicsModel(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, obs_dim),
        )

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        return self.net(x)

class RewardModel(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        return self.net(x)

def train_loop(loader, dyn_model, rew_model, epochs=20, lr=1e-3):
    params = list(dyn_model.parameters()) + list(rew_model.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    mse = nn.MSELoss()

    for epoch in range(epochs):
        total_loss = 0.0
        for obs, act, next_obs, rew in loader:
            pred_next = dyn_model(obs, act)
            pred_rew = rew_model(obs, act)

            loss = mse(pred_next, next_obs) + mse(pred_rew, rew)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()

        print(f"epoch={epoch} loss={total_loss / len(loader):.6f}")

@torch.no_grad()
def one_step_planning(obs, candidate_actions, dyn_model, rew_model):
    """
    Chọn action có predicted reward tốt nhất sau 1 bước.
    obs: tensor shape [obs_dim]
    candidate_actions: tensor shape [N, act_dim]
    """
    obs_batch = obs.unsqueeze(0).repeat(candidate_actions.shape[0], 1)
    pred_rew = rew_model(obs_batch, candidate_actions).squeeze(-1)
    best_idx = torch.argmax(pred_rew).item()
    return candidate_actions[best_idx]

if __name__ == "__main__":
    # TODO: thay bằng dữ liệu thu từ Unity
    dummy = {
        "obs": [[0.0]*10 for _ in range(1000)],
        "act": [[0.0]*2 for _ in range(1000)],
        "next_obs": [[0.0]*10 for _ in range(1000)],
        "rew": [0.0 for _ in range(1000)],
    }

    ds = TrajectoryDataset(dummy)
    dl = DataLoader(ds, batch_size=64, shuffle=True)

    dyn = DynamicsModel(obs_dim=10, act_dim=2)
    rew = RewardModel(obs_dim=10, act_dim=2)

    train_loop(dl, dyn, rew)
```

Skeleton này không phải LeWM gốc. Nó là **world model tối giản** để bạn có thể thật sự triển khai trong Unity project trong thời gian ngắn.

**Nếu bạn muốn thử đúng hướng LeWM research**

LeWM repo chính thức hiện ghi workflow kiểu Unix:

```bash
uv venv --python=3.10
source .venv/bin/activate
uv pip install stable-worldmodel[train,env]
python train.py data=pusht
python eval.py --config-name=pusht.yaml policy=pusht/lewm
```

Dữ liệu đặt dưới `$STABLEWM_HOME`, format `.h5`, checkpoint cũng đi theo cấu trúc đó. Trên Windows, vì instructions công khai đang dùng `source` và `export`, cách thực tế nhất là chạy qua **WSL2** hoặc Git Bash nếu bạn thực sự muốn tái hiện benchmark gốc. Tôi **không khuyên** dùng đường này làm trọng tâm cho demo Unity đầu tiên. citeturn29view1turn29view2turn31view0

## Roadmap thực hiện, dàn ý thuyết trình và kết luận

**Roadmap thực hiện dự án**

| Giai đoạn | Việc cần làm | Thời gian ước tính | Kết quả đầu ra | Rủi ro |
|---|---|---:|---|---|
| Mức 1 — Demo an toàn | Clone game, mở Unity, tạo TrainingRoom, gắn ML-Agents, train PPO baseline | 3–5 ngày | Agent biết đi tới goal đơn giản | Reward chưa ổn, observation chưa đủ |
| Mức 2 — Demo tốt | Thu trajectory, train next-state predictor + reward predictor, so sánh PPO vs PPO+prediction | 1–2 tuần | Có câu chuyện “world model giúp gì” | Predictor không chính xác, integration mất thời gian |
| Mức 3 — Demo nâng cao | World model riêng, imagination rollout ngắn, biểu đồ loss/reward/video | 3–4 tuần | Demo có chiều sâu nghiên cứu rõ | Dễ trượt deadline |

**Dàn ý slide 10–15 trang**

| Slide | Nội dung chính |
|---|---|
| Mở đầu | Vấn đề: AI game truyền thống thường phản ứng tốt nhưng ít “tưởng tượng” |
| Bối cảnh | RL model-free mạnh nhưng tốn dữ liệu; rule-based dễ làm nhưng cứng |
| Khái niệm | World Model là gì, vì sao quan trọng |
| Feynman | Ví dụ nhân vật 2D né hố, né quái, nhặt item, tới cửa |
| Lịch sử | World Models → PlaNet → Dreamer → MuZero → DreamerV3 → LeWorldModel |
| LeWorldModel | Nó là paper/repo chính thức gì, giải quyết collapse thế nào |
| Sự thật triển khai | LeWM hiện là research code, chưa có pipeline Unity chính thức |
| Đề tài thực thi | Chuyển thành: ứng dụng ý tưởng World Model vào AI Agent trong Unity |
| Repo game chọn | Vì sao chọn `2DRogueTest` |
| Kiến trúc hệ thống | Unity scene + ML-Agents baseline + prediction module |
| Demo flow | Agent baseline, agent improved, biểu đồ reward/success |
| Kết quả / kỳ vọng | Agent học tốt dần, tới goal nhanh hơn, ít va enemy hơn |
| Hạn chế | Chưa phải LeWM pixel-JEPA đầy đủ; mới là world-model-inspired demo |
| Hướng phát triển | Thu trajectory nhiều hơn, visual observations, Dreamer/LeWM-style pipeline |
| Kết luận | World Model là hướng rất mạnh; demo Unity khả thi nhất là bản tối giản có baseline |

**Demo flow nên làm trong buổi thuyết trình**

1. Mở game gốc để chứng minh repo clone/run được.  
2. Chạy agent **Heuristic / random** để người nghe thấy agent ngu.  
3. Chạy agent **PPO baseline** sau train để thấy nó biết mục tiêu.  
4. Chạy bản **có prediction module** để nói “đây là bước đi về phía World Model”.  
5. Chốt bằng 1 slide so sánh với LeWM gốc: “LeWM pixel-based JEPA, còn demo em là world-model-inspired để phù hợp Unity và deadline.”

**Những câu hỏi giảng viên có thể hỏi và câu trả lời mẫu**

- **Hỏi:** “LeWorldModel có phải framework Unity không?”  
  **Đáp:** “Không. Sau khi kiểm tra paper, website và repo chính thức, LeWorldModel là một research world model theo hướng JEPA, dùng benchmark offline như TwoRoom, PushT, Reacher, OGBench-Cube. Em dùng nó làm nền tảng khoa học, còn demo Unity của em là ứng dụng ý tưởng world model.” citeturn30search0turn31view0turn41search1turn42view0turn42view1turn42view2

- **Hỏi:** “World Model khác gì PPO bình thường?”  
  **Đáp:** “PPO baseline học trực tiếp policy. World Model học thêm cách môi trường chuyển trạng thái, nên có thể tưởng tượng tương lai rồi hỗ trợ quyết định. Dreamer là ví dụ điển hình của hướng này.” citeturn11search0turn32view0

- **Hỏi:** “Vì sao em không dùng LeWM nguyên bản?”  
  **Đáp:** “Vì release công khai hiện thiên về benchmark research offline/HDF5, không có pipeline Unity chính thức. Nếu cố ép vào Unity trong thời gian ngắn thì rủi ro demo hỏng rất cao.” citeturn29view1turn31view0

- **Hỏi:** “Demo của em có phải World Model thật không?”  
  **Đáp:** “Baseline thì chưa. Bản mở rộng có next-state predictor và reward predictor đã là một world model tối giản. Nó chưa phải full LeWM/DreamerV3, nhưng đúng tinh thần model-based decision making.”

- **Hỏi:** “Nếu có thêm thời gian em sẽ làm gì?”  
  **Đáp:** “Em sẽ chuyển từ vector prediction sang pixel prediction, thu trajectory lớn hơn, rồi thử Dreamer-style imagination rollout hoặc pipeline gần hơn với LeWM.”

**Open questions / limitations**

- Tôi **không tìm thấy** tài liệu chính thức nào cho LeWM tích hợp trực tiếp với Unity hoặc Unity ML-Agents trong các nguồn chính thức đã kiểm tra.  
- Tài liệu ML-Agents hiện có **một chút lệch pha** giữa installation docs, GitHub releases và package numbering, nên bạn nên pin chặt version theo setup mình chọn. citeturn16view1turn17search1turn36view0  
- Với repo game Unity open-source, một số README và `ProjectVersion.txt` **không hoàn toàn khớp nhau**, nên luôn kiểm tra `ProjectSettings/ProjectVersion.txt` trước khi cài editor. citeturn24view3turn25view0turn21view5turn25view3  
- LeWM gốc là **pixel-based offline goal-conditioned planner**; demo Unity mà tôi đề xuất là **phiên bản thực dụng hơn**, thiên về model-based game AI hơn là tái hiện paper 1:1. citeturn41search1turn29view1

## Checklist cuối cùng

- [ ] Đọc hiểu World Model cơ bản  
- [ ] Hiểu rõ `LeWorldModel` là paper/repo nghiên cứu, không phải framework Unity  
- [ ] Chọn repo Unity mục tiêu  
- [ ] Cài đúng Unity Editor version của repo  
- [ ] Clone được game Unity  
- [ ] Chạy được game  
- [ ] Cài được ML-Agents  
- [ ] Tạo được scene training riêng  
- [ ] Gắn được `Agent`, `Behavior Parameters`, `Decision Requester`  
- [ ] Train được agent baseline  
- [ ] Xem được TensorBoard / reward curve  
- [ ] Xuất được model inference chạy lại trong Unity  
- [ ] Thu được video demo  
- [ ] Có ít nhất một biểu đồ reward hoặc success rate  
- [ ] Có slide giải thích World Model bằng ngôn ngữ đơn giản  
- [ ] Có slide nói rõ LeWM gốc khác demo Unity ở đâu  
- [ ] Có phần nói về hạn chế  
- [ ] Có hướng phát triển nếu được hỏi thêm

**Kết luận ngắn gọn**

`leworldmodel` hiện tại **là LeWorldModel thật**, một paper/repo chính thức năm 2026 về **JEPA world model học từ pixel**; nó **không phải typo** và cũng **không phải framework Unity sẵn dùng**. Nếu mục tiêu của bạn là **hiểu sâu + làm demo thuyết phục trong Unity**, chiến lược tốt nhất là: **dùng LeWorldModel làm lõi học thuật**, nhưng **triển khai demo theo hướng World Model đơn giản trong Unity** với một game open-source dễ kiểm soát, mà lựa chọn tôi khuyên mạnh nhất là **`2DRogueTest`**. Lộ trình đáng làm nhất là: **ML-Agents baseline trước, sau đó thêm prediction module để tạo world-model-inspired agent**. Cách này vừa trung thực khoa học, vừa khả thi kỹ thuật, vừa đủ mạnh để thuyết trình tốt. citeturn30search0turn31view0turn41search1turn22view7turn25view2turn16view0