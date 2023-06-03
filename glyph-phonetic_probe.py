# coding:utf-8
import argparse
import collections
import os
import pickle
import random
import traceback
from pathlib import Path

import opencc
import pypinyin
import torch
from hanzi_chaizi import HanziChaizi
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from utils.utils import mock_args

hanzi_list = list(
    "的得地他她它哪那真这么吗啦一国在人了有中是年和大业不为发会工经上市要个产出行作生家以成到日民来我部对进多全建公开们场展时理新方主企资实学报制政济用同于法高长现本月定化加动合品重关机分力自外者区能设后就等体下万元社过前面农也与说之员而务利电文事可种总改三各好金第司其从平代当天水市提商十管内小技位目起海所立已通入量子问度北保心还科委都术使明着次将增基名向门应里美由规今题记点计去强两些表系办教正条最达特革收二期并程厂如道际及西口京华任调性导组东路活广意比投决交统党南安此领结营项情解议义山先车然价放世间因共院步物界集把持无但城相书村求治取原处府研质信四运县军件育局干队团又造形级标联专少费效据手施权江近深更认果格几看没职服台式益想数单样只被亿老受优常销志战流很接乡头给至难观指创证织论别五协变风批见究支查张精林每转划准做需传争税构具百或才积势举必型易视快李参回引镇首推思完消值该走装众责备州供包副极整确知贸己环话反身选亚带采王策女谈严斯况色打德告仅气料神率识劳境源青护列兴许户马港则节款拉直案股光较河花根布线土克再群医清速律族历非感占续师何影功负验望财类货约艺售连纪按讯史示象养获石食抓富模始住赛客越闻央席坚份士热限米银息校均房周游千失八检足配存九命尔即防钱评复考依断范础油照段落访未额双让切须儿便空往你层低奖注黄英承远版维算破铁乐边初满病响药助致善突爱容香称购届余素请白宣健牌促培竞巴稳继紧字困刘旅声超随例担友号显却监材且春居适除红半买充陈火搞图阳六察试太什执片古七球修尽控讲排粮武预亲挥卖审措荣洲卫希店良属险曾围域令站苏龙念罗吨器汇康减习演普田班待星飞写矿轻扩言章汽靠毛终仍景置底福止离泽波兰核降训逐票菜座献钢眼损宁像苦印融独湖早予夫编换欧努著顾征升态套介送某斗状画留航派室临兵补宝略黑综云差纳密贫剧犯阿击遇岁阶烈督吃丰馆招害官树听庭另沙私针胜贷网愿托缺园假酒音巨既判输讨测读洋括筑欢刚庆久陆找楼激晚绝压故互签汉草木亩短绍迎吸警藏疗贵纷授登探索湾宏录申诉秀序顺死卡歌午孩桥喜川邓扬津温船库订练候退违否彩棉帮拿罪币角召灾妇杨奋绩虽煤免笔够永圳停奥鲜朝吴岛觉移尼急博贯拥束左细舞幅语俄奇般简拍脑债固威券追筹刻映繁伟甚饭右彻烟沿街血冲洪植誉刊玉厅救潮迅伍怎付倍顿述播励斤乎纸振旧障鼓艰呼吉男绿尚夏亏季松哈祖典韩遍夜轮板抗摄杂皮贡借幕罚伤岸扶乱曲脱践危澳童散味叶累谢孙邮雄兼微呢谁惠偿署择染答块徐鱼赞课盛延瑞怀堂驻零辆齐胡途封似润守毕坦母雨败朱污趋械纺租灵拓残含握跨衣储瓦蒙鉴析竟骨档秘禁赵宾异伊智钟键辉跃冷倒庄毒仪涉泛宗鹏归岗雷礼尤休泰疾肥珠叫牛宜抵挂寻父攻佳塞架符裁虑肉启丽露鲁秋昌估射册若宽厚盾硬末轨饮勤茶诗郑冠涨篇泥唱纯坡熟浙晓抢丝锦载笑勇杰患乌坐雪戏背塔翻沈遗聚渠哥享迹森辽衡掌牧附操赶览野盟殊仁错萨夺梅误词董潜卷矛腐亮冒盖旗井凡震峰坏倾距壮惊盘梁摆径忠冰峡丹避珍乘刑扎透迫箱莫跑穿祝乏厦渐软询折浪朋敢诚弱疑邀沉端床络疆缩脚甘贴勒荒唐静缓侵句尊塑肃怕耕痛援劣伙挑洗暴冬龄乔餐肯廉跟阵伐悉忘闭奔恢宋泉杯渡奉婚赴恩盐掉洁亡洛聘蔬混摩抽鸡剂胆麦谋雅废贺羊阔唯捐返隆穷辛猪帐饰郭颁灯绕诸伴顶祥谓恶番敏旦劲缴麻屋跳码鞋扣迈忙趣盈棋勃敬辑摊旺纠炼梦偏渔牙侨黎赔裕宫谷概稿柱弹殖秩凭拨幸洞伪沟姓遭涌陶迁诺拔畅忧胞丁蓄贝舍腾杀煌圆伦横薄畜毫豪弟呈佛邦您墨徽惯循蓝烧触陕拖伯盲宪净卢炭籍秦粉妻爆欣释玩俊欠蛋猛迪苗暂貌遵锡楚桂昆杜皇醒燃凤截铺液撤胶慢杭虚辞曼毅咨俗糖忽芳姐耗妈谊浦频阻允宅窗默胀弃倡灭甲症埃滨赏莱拒淡坛陵绘虎竹赢锋篮迷纽轿贩递娘圈挖炉替幼乃郊颇戴滑徒崇涛焦凝墙吧炎刀玻寿履圣昨酸朗媒桑铜仲亦诞揭纵漫愈辟赠旱奶泳枪骗虫池镜浓拆艾扫娱钻碍寒迟邻曹盗穆豆赚晨浩彭耳瓜扭脸燕摇寄仿炮晋泪欲饱壁锁刷柬诊磨捕寨滚膨孔添帝辖炸旨吁址驶抱嘉拜扰袋佩阴辈锅赖剩押怪浮枚栏毁柳恐敦孟旁仓岩伸岭耐懂捷璃溪暖纤汗疫巧旋侧冶陪鸣瓶纲挤旬舆喝陷缘稻饲滩隔慰朴隐灌拟偷闲赫恰慧蒋闹邹牵柴刺滞彰俱勘填琛尝贾搬淮奏荷滋覆役秒踏巩摸荡辅惜柜肖颗搏氏姑弄姜君舒兑宇割哲摘钦逃漠忆敌宿啊凌耀闯阅贪赤汪悲抑瓷冯厉粗菲琴堡斌掘稀衰驾雕牢氛驱妥悄郎巡臣羽灰癌颖姆漏袭贤鸟暗茂孤惩榜袁桌卓傅剑堆兆狠轰拳妹绒裂潘兄洽叹涵贿侯岚熊绪阁尾碑尖腿涂栽坝犹铸肩闪诱辩芬睡奠伏妙乙绸廷夕恒梯赁霞攀枝译描湘磁吕硕爸肝峻葡衷搭唤薪挺逝狗蔡宴蓬撞铝牲舰胁崛桃斜丧烂屏砖墓详逾函跌抚插戈凉啤脉滥赋柏堤腰泊寺尘蒂削仙踪冻汤睛艳荐劫框廊惑页拼堪携丈乳挪谱舶埔遥菌塘氧晶洒株颜虹岳胸忍甜匹瞩懈爷丛莲叙鸿逢抬嘴弘炒喷吊窝衔吹霸仔垦胎慎脏歧疏悠慕漂杆萍舟吐玲凯戒盼偶盆慨弊箭茅衫罐串辐腹钩碰昂酬晰姿彼锻飘嫁竣缝蹈悬紫浅缆喊昔驰湿剪侦坑姚魏扑挣焕皆狂泡骤堵膜禽锐芝帽擅沪晤婆埋劝碗玛顷鸭娃豫匆魂哭庞亭屡逼尺撒鹿讼弥坊碎缔霍壤萄铃稍丘肿烦苹庙雇汛孝辰吞汰怨酿耶咱欺丢琼棚披渴屈弗疲帕昭盒仰萧牺撑抛鼠纱翼兹骑糊契铭淘顽撰乒淑妆窑柔姻苍谨卿灿栋敲窃菊郁催眉邱揽鼎韦肤娜俏呀拚寸爬悟尿罢圭葬聪沃肠厕慈恋绵橡圾垃翁粤脂歹憾阐甸巷蜂轴艘垄衬阜惨冀幽厘崭筋寓迄渗碘碧赌袖奈崔悦捞剥孕逆婴脆缅艇谭笼儒粒诈遣垂磋卸帜枣幢淀帆蛇宰殿猎叔夹帅沧魅俩牟钓葛罕渤汕溢擦袱嫩桶殷酷呆卧暑骄幻囊掀醉牡饼扇蒸赣俭椅枢彦樊吾仗彬砂绳巾喀勋愁碱谦壳轧潭浆挽邢啥焊钞烤廖猫狱腔喻御蕴坎魔刮瘤茫竭莉链淫愤纹咸睐睹裤夸滴雾搜拘龚凶茨傲鞍鹤蚀颈翠卉汁冈狮隧弯胃沛募琳疼蚕泼磷捧炳绣朵涯掏奎聂孜韵浑翔魄掩斥敞腊愧粘丑溉斑柯谐烯禄浴涝鬼薛瘦挡昏鹅湛逻虾沂辱叉鼻厨鲍鞭辣潇乓肺尹颂邵澜桐鹰妨闽屠畏翰塌亟寂赂犬聊暨垫泄漆旭蕾坪涤挫佐瞄拦硫棒杏爽碳畔熙襄祸乾淹臂莎辜阎庸砍捉勾垒衍坤噪毯倪扮铅遏哀愉瑶咬嫌闸恳齿杠怒兽浇肇鄂溶哄棵盯梨灶屯狭陋啡浸淋濒脊戚勉膏氨墅沸挨蔓抄芒秉刹饶厢咖魁骚缚遂恨跻螺辨菇帷凰椒汝瞬淄舱馈桩炬誓卜麟岂兔眠泵拐肚匪芦匈霉蜜荆雁窄秧枯仆嘱壶谅哨肌贬叠稽岐沫肆醇菱彪躺摔膀甫逊凑渊喂藤砸悔杉霜厄忌桔筒丙臭拾芜禹丸蟹嘛俞翅尸澄骂睦馨郝贮陌钧轩赃笋歉逸歪巍萃崖窟踢锣萎庐剖籽甩饥苑恼渣痕莞硅晴巢瘫缠隶筛穴昼埠宠肢饿仑逮兢趟糕妮邪抹俑萌匠扔酱葱礁掺雀髓悼挚蔚枫庚伞侃僵捆蒜溜傻蔗谜斋蝶沾闷驳耿槽黔吓肾芽栗朽荫榆皖曰徊奴迭僻蓉靖氟滔羡愚尧俺徘罩磊镑舌曙纶粪匙钉佼扯踊躲猴纬咽酝挠宛瑰歇抒茧穗祭鑫趁痴裙猜耘碌锈晒潍弦稼狼拢梧芯眷哑宙厌逛谴邯呵蜡寥钥耸媳熏蚁惕颠娟亨吟蒲梭瞻渝喉遮慌夷韶焰尉珊胖蕉粹裹琦秽侠奸挝绑曝棍婉镶熬傍燥氯骆晃鸽疯琢聋瑟暇绥禅溃腺垮阀撼煮佣滕淤蹲栖硝睁荟荧抖坟芭臻锭晖倦倘喘邑锤惧荔毗觅矮恭钙氮缸瞧颤萝佑怡瘾寡烹摧棠缪雏韧喇兜坯坷贞仇缉帘竖糟猖懒凿洼喧谣驼烫锌椰崩沥汾磅霖棘扛彗矩瞒陇绎诫斐卵铮钾宵簿秤畴斧擂剔躁冤讳寅焚漳鳖哺耻僧琅粟怖咏蜀淳柑缕烁氢蔽琪泣阮镀殴虞虐炊搁诀掠坠屿髦酋躯吵遐寞仕稚僚楠矶筝彝叮熔槐潢芹郸匾咋玄裔陡哗怜襟刃脾嵌拱慷痪跋孚峪钊滇苟晕墩膝羞乍腻詹讶敷肴莹衢柿朔袜枕烘匀歼泻樱吻翟堰苯隙娇獗汲蛙斩靡沁乞姨翩沼嘎畸矫骏薯绚窜藻矗皂楷腕篷徇耽娼犁榻茄棕汹峨蹄昧奢涩灼踩粥拣旷簇溯攒沓呕梳搅砌纫渭澡撕漓葆辍肪祁鞠蛮捏诵娣岱瀑啸裸鸦瑛躬舜忱豹纂恤惟赐俯犀媚嫂嗓蚊茬驭缀皱凳钮蚂姬扒嫖跪凹揣尬沦尴豁玫殡淌叭唇啃裘卑琐矢拯忡勿盎茵椎脖拂骅葫迢薇龟绞眶沐傣浊舅叛浚窘栓酶笛泌榄惹铲碟捡恪酯滤匿酵砚贼匮熠鳞麓镁氓苇廓巫踵竿蘑翘梓贻鳗帼冉泓狐涟崎窍瑜讽逗铎掷璀泗浏陲醋苛攘璧瀚哩暮矣蚌悖扼漯烛蝴屑墟俘侣庇陀煎秸弓捣譬炜炯拌扁彤锚禾侮秆绮嚣樟咐枉窦桦寇哉狸耍馒驹隋冕疮咄妄峙娄溥腑钠栩糙滦呐鲻娶祺刨褒橙茹谎抉慑媛橄戎迩雯璨雍惶扳桢霓账梗炕裴韬杖痹缤沽燎煞删辙爵缭劈烨槌媲凛莆颅锯膳澎坞瓣婷絮酌涡唁秃禺膊棣芸忻炽榨篆憨戍圩爹蹊饪胺贱睫蝇惫拇赈泾盏弧剿硒毓皓菏灸湄炙祠荻捍嚼朦屹紊藜驴寝兮隘祈榕臧蝉绢瞎闵鳌娥藉娅烽楂摒凄凸熄孵叩渎胳匡袍卒怠桓莽倩泸藕陨辗骋峭冥饺亢圃颐擒铵鳄簧愣璜钰拙瘠靳隽罹岑镭榴恕毋囤汀绽窖筷擎猿诲碾夭筐邃藩诬芙胚哇垣胧帖殉毙壑绰憋亥涅屁璞缮侍倚稠棺棱葵诣笨橱寰郡垢徕眺胰谆窥霄栉舸蹦坂瞪珲釉跤挟侄肘嘲刁缎嚷痒敛祛绅孰痫闺椿噶恍伶峦酥萦苎癫涪锲蜚拎嵩昊娴涣烙璋笃囚祯篱讴舷纭锄巅卦摹眸柄踞焉辄褚褐湃夙堕岔惦疚谍奕羚帧澈濮捎漾吼锰趴菩簸仃渲札谙咕桨咀郴咳呜蛟拧莘驯庵弼逞蹬姥撂镍晏疡爪骥楞钳懋寐淇琉杞菠铨翌靶侗瑙馅丐痊娓侈苓聆睿偌釜噬曦燮哟瑾瞿璇拮憬鹊勺憧嗜啼檐柚呱渍镌妃溺鸥粕沱榭隅毡禧瞅鲸淆阪茁渺瞥茜瘟礴伺谛锹蔼虔莺迸磕赡泱栈甄镐抠嬉诿甬绊饵谬梢颍揪琶褥佟腥辊溅琵鄯拴喃笙酰粱卤芮膛斓潼鸵侥讷婿吆羁嗣蜒栅疙拷戳镛芷钛蜿铀夯摞雌酣荼蝎锥姊瓢祀玺弛犷哦茸鱿绷茎惋亘珑莓掂迥鲤殃瘩叨螃奄腈疟沭钨昕膺涿糠氰揉狩檀悍缫哮衙瑚潞谤搀洱涓袤痰乖冗芋甭骸幌涮俨敖槛狄牒恺雹赎庶熨蛛佰蓦鄱煽腌黯疤倔剌斡诽锵筱妍掖铿脐捅弈邸湟眯赦拄啪玮轶蛾麋炫赊靴箔菁撬裳戌缨蝗撇奚瀛噩怯蓓匕咚瞰佬泞扉皋晾麒姗跚瘀鄙猕拭鲟祷脯砺驿陛瘁搓舵汞哼胫珀邬磺馏馍铢诧涧吏苔潺邳烷囿斟滁殆酚狡孺恬沅铬湍啧囱蒿鹃柠漱胥妖洙珂茉蹒圻鬓搂葩佘渥诙袒捂瞠妓铐澧袂馁汐匣逍谚窒蔑糯汶壹岖盔嘘迂嘀锢讥吭抨屎獭褪咫稷迦檬塬蠢蓟咎皿驮俐坍惭垛鹭鸾蹴撩诠恙臃遨睬踌浒搪郧竺翡宦冽憩萱拽卞槟躇蘸肋呛濡酮眨撮矸垸蛀黛涸脓徙撷曳峥渚镖钴骊袅磐掣沌埂嘿琏楣豚诡悸麝煦矾羲唉溧呻覃兖吱惰羹钝枸姣颓铣梆骇淅孢叱谧泯谟恃薹筵鏖栾鹜哽掬辘茗瓯绛筠铤袄殚梵挎遴榈蜕癣垠厮幄偕焱攥裨炖旮旯蔺骡娩伫猝窿虏屉缜咒筏骼璐剃涕猗淼侬阙嗅鸳嘈霏珩沮捺硼荃驷漩嘻眩掰伽脍婪煜鹄壕崂翎痞兀婺鸯楹咤徜嫉篓烃铂咪掐匝杼蕃箍荤砾嘶皑宕荪哎汴貂邡淦蕙弩堑惬偃徉箴赘啻凋穹酗憎芥唾闫晔苞昶甙笺吝蕊鳝衅猩薰昱趾淞坳怅翱汩琥岌阑粼羌霆篡塾酉裱韭唠廿闰攸黝蛤厥荞瑕柘祚疵愕蕨牦飨疹嗷癖芪漕隍徨逵泠嵘嗡岫岷擞陂颊咔卯婶椭惘歙幺臆叽缰睽勐暄弋痔秭煲琮嘟犊玖怦丕溴罂瓮丞惮癜晦攫镰镯柞舫铆蹼妩熹铱褂丫笆妒噢噙琬冼荀蟾捶嗒町嫣肮皎旖恣钚砥吩茯馥钎甥嗦蜗浔谒辫亳彷珏咯淖妊佤玷嘹崴於辕贲扈伎旎孽耙娠戊冢跷砷焘羔圪耄钼悻荥唑稞邝莅杷醛嗽唆拗碴馋胱琨茏糜懦骞蜘嚓怵抡唢腆涎灏臼墒暹椽牍钒猾榔懵枇樵锶籼箫漪帚钵赓捻郅儋烬锂剽锑鄢鄞臾喳胄耋阱笠瓴啬杳萤莠嶂浜傩遒轼睢倜矽仉唬旌酪腼罄嬗畲祟桅悴讹憔龋嵊绶邕忖箩咆晌愫猷帛麾莒觑吮蟋庥懊阂蒯阡腮潸晟蟀臀罔骁崽绉粽忿肛蠡遛蜓煊蚜坻滹銮悯鼐撵噼忐湮侏粳矍铄坨铉盂锗阖溟俟忑赝鬃敝宸哆靓揩瘸鲅篝氦嚎浃缙飚锷癸柩蛎濂榷鲨钡盹鲫诘诩迤桎遁尕梏楫赳飒锃雉怆痼劾痢喽霹昙畹胭佚狈瘪姹吠铧谏雳咙畦荠娑褶忏惚痉橘漉诏呗晁惆砀馄戟峁昵拈蠕虱洵鹦蛹铛挛倏澍濉钅噜咛俳磬蜷霎肽砼聿怔砭谌箕蹶孪蔷糅挞饨惴禀淙哒枷楝闾蜻嗖淬垩矜郗蚤嫦喋镉饯髋潦镂簌偎鹉岙踱诃籁宓膘飙涞耆荏渑豌琰俎绌埭幡赅锆崮碣珞腋滢蓖伉馗聩幔锨蓥鹑砝酩枰鞘苋粑蹭倌犟俪嶙砻嵋滂葺苒枭翊婀飓阚喟傈藐蜃怂稣亵诒蜇岜霁瞌沏卅舀鹌俸嵇蟒汨砰鞣唏陉佯恿竽瘴祉焙诋濠螂叻垅谩朐稔芍瞳惺萸盅啄眈偻爿蟠炔垭噎蛰擘锏茭悌喔谑峋妪恽韫褓镳饽杈戛鸠萋襁榫霭苄跺杲嗨珉哌娆孀恸缄夔佗饷苷郜鼾颌訇谲溘咧褛逄颦洮逶嫡蠹碓烩醴栎鎏瓤伢蔫怿甾摈畈镣螨秣搔盱痍搐蹉佃绂疽骝霾悚缃懿咂奘轱邗蚝瘘醚湎瞑掮羟仨砣郢砧鳟跛踝轲窠郦踉躏戮篾骐鳍蹂郯跎倭诅鄄褴阆缈嗯妞沤跄箐苕窕楔饴峄腴圄谕揍踹罡佝颔觊篑鲢綦妾镗啕蚬窈揖眙蟑诛钗绯讣睾媾嗬祜镢囹苜坭蛐髯搡叟蹋觎捱碉呋罘荚鹫岿寮扪焖狞鳅嗄嗤擀痂嗟颉蚧儆锴龛嗑锟俚枥懑讫橇嗪虬跆骧陟灞恻涔酐鸪牯钜萘鲶缥曜蚓诤埕墀麸蝠蛊遑厩趄沔耦疱匍揿蚯讪唰舔呷蓿鹧膑刍耷鞑裆趸孑鲲绫埝嘭舢鸢螯吡蝙疸匐桁铠羸鲈囵唛仫庖劭郓骜粲峒腓鹳鳜蚶囫茴峤蟆蘖癯纾僳皙隰缬馐谪捭汊碜塍艮睑狍苫篦蜍锉沣诰晗喙麂謇蹇觐啾踽邈壬燧娲猥歆镒茔昝赭狰孳哧舛噔鹗蚣逅洹腱锒纰蛆蕤姝邰纣嘣钹衩婵孱蹿鲷萼椁浣镓遽赉趔蕲剜邂仡氤獐幛俾铋嗔茌氡诂豢桧畿倥捋仞忒疃浯蜈榛偬稗菖鲳厝踮叼痱貉玑婕琚疴掳钤垧氵黠跹怏揄氲铡濯芾笈崆钕菽隼傥仝囗芗埙簪暧桉镝蚪蜉藁笳菅龃喹橹抿啮蹑逖唔樨巽揶黟訾钣嵯凼恫掇剁珙沆噱揆耒铌泅疝葳隗滟龉钺殒蒡觇黜澹酊垡奂珈濑馕馊嚏痿岘氩茱滓焯抻豉敕掸碲靛摁淝鳏盥皈鲑颢犄翦铰椐胯屺邛庹猬蓊骛浠桠胤鸩痣蛭噌杵啜靼啶煅枋觥毂刽蝈蘅芨戬醮疖忾骷洌呤荦觞谡瀣蝣糌倬碚蹙痘砘绀虢蕻肓蛔唧桀蝌侩棂樯挈轫巳崧蓑藓鳕瑗帙馔豺痤郇殓髅轳逯嗫戕嚅蛳琊嘤疣蚱钯钿碇咣毽迳喱逦廪邙囡匏扦亻咝凇纨涠庠溆醺炀烊肄龈谀锱瘢枞皴贰晷闳斛屐讦婧苣蔻绺渌瑁螟叵颀穑膻羧螳绦誊蜥楦恂靥咿翳瓒枳啭樽嫒婊搽铒跗凫菡篁髻裾栲癞蓼氖孬喏砒姘衽缛嵬挹缢慵呦箸蹩槎榇舂嗲胴谔岢圹娌潋蛉酃鲵鲇娉亓碛芊忪谇笤韪勰呓俣圜愠仄炷毖筚伧棰磴滏篙肱笕堇馑荩榘哐傀崃罱痨儡鹂檩垴仵檄芎阉刈壅馀庾妯躅獒阊笞饬钏硐椴泔硌鹘鳇豇狙戡莨啉辇臬殇舐黍薮眭佻嗵煨莴蚴妤瘐擢蛏蹰龊辏绐氘骶莪珐缟聒讧岬胛桷谰戾撸鸬雒嘧囔铍骈掊茕噻铯柁艉龌硖罅魇酽咦嶷羿轸趵荸薜踟玳啖蔸槁鲛疥砬唳弭曩黏镊泮霈淠柒颧瘙痧辋郄燹泫郾鹞钇殪痈甑踯翥婢檗柽啐菪嶝腭嗝剐笏蟥戢阄噘撅尻贶辚蜢颞忸胼阕竦焐揠邺鳙啁稹徵诌隹舨哔卟伥苌鹚箪缍锇蝮诟洄浍诨犍硷噤垲郐椤嫫伲脲殍噗溱箬厍钽钍恹鬻爰砦蓁胝颛褙鳊邴铖镫腚钭颚鲂悱狒佶偈堀绔醪坜疠椋犸暝佞哝瞟荨芩逡溽裟挲抟暾崦芫荑薏莸欤栀斫镞嗳鸨跸骠俦谠簟棼驸掼倨橛犒邋耧蝼虻铙郫汔诮楸阒绻叁臊钐腧闩菘阗忝橐翕阋踅窨鹬鼋樾錾吒旃弁侪坼蚩嘬糍骢氐呃榧玢绋蚨钆岣菰罟嘏埚绗嚯藿笄袈羯肼暌啷蒗蜊獠鬣熳黾乜镆怩驽旆髂仟芡谯恁鳃艄莳艏趿遢鲐醍僮氽刎芴喑墉昀箦鄣摺钲贽缵鏊锛瓿廛瘳亍遄褡垌椟酆砩桴赙坩臌曷跽湫榉黧猁钌镏缦殁赧埤悭缱衾鲭铩猞眚铈谥耜飕饕餮骰乇绾鹇鲞爻蜴镱铟莜祗濞镔逋谄谶酲茺樗憷莼撺柢阏砜垓旰妫衮嗥郏鞯徼孓钪侉夼跬铼嫘蟊茆睨怄蹁谝嘌綮嫱筇犰穰铷筲哂炻豕秫笥涑铊帏闱鋈舾屣狎哓噫璎铕宥阈豸辎趑龇捌秕荜愎窆镲谗踔苁酢呔聃镦屙鲱鬲膈铪醐獾鲩虺葭牮礓苴讵颏裉诳栌氇镙哞柰袢帔睥苤嫔笸氆佥箧跫蚋鲥扌狲桫溏铽殄脘洧肟绡咻洫癔洇嵛磔胗肫赀眦吖瑷埯畚妣飑豳髌砗铳楮蔟毳锝堞疔葑缶菔疳彀胍磙顸薅翮猢怙蒺廑妗髁醌粝魉旒蝥缗衲呸醅芘蚍圮榀萁苘逑诎劬蕖朊剡蟮椹饣酞帑葶菟魍庑葸氙谖鞅狺夤嬴瘿饔雩鹆橼赜潴骓缁诹怍杓艹檫媸氚呲殂矬笪迨纛簦玎苊轭匚鼢呒缑诖炅鲧唿戽鬟恚袷瘕枧洚桕雎蠲剀诓瘌镧铑鳓蓠呖跞裢裣埒捩鲮熘嵝瘰镘脒")
hanzi_list = hanzi_list[:3000]

def load_obj(filepath):
    with open(filepath, "br") as f:
        return pickle.load(f)


def save_obj(obj, filepath):
    with open(filepath, "bw") as f:
        pickle.dump(obj, f)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def is_chinese(uchar):
    if uchar == u'\uf7ee':
        return False
    return '\u4e00' <= uchar <= '\u9fa5'


class PhoneticProbeDataset(Dataset):

    def __init__(self):
        super(PhoneticProbeDataset, self).__init__()
        # Get all chinese characters.
        tw2zh = opencc.OpenCC('t2s.json')
        chinese_chars = set()
        for token in hanzi_list:
            if is_chinese(token):
                chinese_chars.add(tw2zh.convert(token))
        chinese_chars = list(chinese_chars)

        # Create chinese pinyin list of chinese characters.
        chinese_chars_pinyin = []
        for char_ in chinese_chars:
            chinese_chars_pinyin.append(pypinyin.pinyin(char_, style=pypinyin.NORMAL)[0])

        # Create Positive samples.
        positive_samples = []
        for i in range(len(chinese_chars)):
            pinyin_i = chinese_chars_pinyin[i]
            for j in range(i + 1, len(chinese_chars)):
                pinyin_j = chinese_chars_pinyin[j]
                if pinyin_i == pinyin_j:
                    positive_samples.append(((chinese_chars[i], chinese_chars[j]), 1))

        # Create negative samples.
        negative_samples = []
        while True:
            i = random.randint(0, len(chinese_chars) - 1)
            j = random.randint(0, len(chinese_chars) - 1)
            if chinese_chars_pinyin[i] != chinese_chars_pinyin[j]:
                negative_samples.append(((chinese_chars[i], chinese_chars[j]), 0))

            if len(negative_samples) == len(positive_samples):
                break

        # Merge positive samples and negative samples and then shuffle them.
        dataset = positive_samples + negative_samples
        random.shuffle(dataset)
        self.dataset = dataset

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)

class GlyphProbeDataset(Dataset):
    chinese_chars_components = None

    def __init__(self, cache=False):
        super(GlyphProbeDataset, self).__init__()
        cache_path = './cache/GlyphProbeDataset.pkl'
        if cache and os.path.exists(cache_path):
            self.dataset = load_obj(cache_path)
            print("Load GlyphProbeDataset from cache.")
            return

        # Get all chinese characters.
        tw2zh = opencc.OpenCC('t2s.json')
        chinese_chars = set()
        for token in hanzi_list:
            if is_chinese(token) and len(token) == 1:
                token = tw2zh.convert(token)
                if is_chinese(token):
                    chinese_chars.add(token)
        chinese_chars = list(chinese_chars)

        chaizi = HanziChaizi()
        chinese_chars_components = GlyphProbeDataset.get_chinese_chars_components()

        positive_samples = set()
        for u in tqdm(chinese_chars_components, desc="Init Glyph Dataset"):
            for w in chinese_chars:
                if w not in chaizi.data:
                    continue

                w_components = chaizi.query(w)
                if u in w_components:
                    if not is_chinese(u):
                        continue
                    positive_samples.add((u, w))

        negative_samples = set()
        while True:
            u = chinese_chars_components[random.randint(0, len(chinese_chars_components) - 1)]
            w = chinese_chars[random.randint(0, len(chinese_chars) - 1)]
            if not is_chinese(u):
                continue

            if not is_chinese(w):
                continue

            if (u, w) not in positive_samples:
                negative_samples.add((u, w))

            if len(negative_samples) == len(positive_samples):
                break

        positive_samples = [(item, 1) for item in positive_samples]
        negative_samples = [(item, 0) for item in negative_samples]

        # Merge positive samples and negative samples and then shuffle them.
        dataset = positive_samples + negative_samples
        random.shuffle(dataset)
        self.dataset = dataset
        if cache:
            mkdir('./cache')  # FIXME
            save_obj(self.dataset, cache_path)

    @staticmethod
    def get_chinese_chars_components():
        if GlyphProbeDataset.chinese_chars_components is not None:
            return GlyphProbeDataset.chinese_chars_components

        chaizi = HanziChaizi()
        chinese_chars_components = set()
        for values in chaizi.data.values():
            for value in values:
                for component in value:
                    chinese_chars_components.add(component)

        GlyphProbeDataset.chinese_chars_components = list(chinese_chars_components)
        return GlyphProbeDataset.chinese_chars_components

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)


def create_dataloader(args, collate_fn=None):
    # dataset = PhoneticProbeDataset()
    dataset = GlyphProbeDataset()

    valid_size = int(len(dataset) * args.valid_ratio)
    train_size = len(dataset) - valid_size

    if valid_size > 0:
        train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size])
    else:
        print("No any valid data.")
        train_dataset = dataset
        valid_dataset = None

    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              collate_fn=collate_fn,
                              shuffle=True,
                              drop_last=True)

    if valid_dataset is None:
        return train_loader, None

    valid_loader = DataLoader(valid_dataset,
                              batch_size=args.batch_size,
                              collate_fn=collate_fn,
                              shuffle=True,
                              drop_last=True)

    return train_loader, valid_loader


#####################################################


def load_model():
    from models.MultiModalMyModel_SOTA import MyModel
    model = MyModel(mock_args(hyper_params={}, device='cuda'))
    model.load_state_dict(torch.load("./temp/multimodal-sota.ckpt")['state_dict'])
    model = model.to("cuda")

    return model._tokenizer, model


class GlyphPhoneticModel(nn.Module):

    def __init__(self):
        super(GlyphPhoneticModel, self).__init__()
        self.tokenizer, self.encoder = load_model()

        self.cls = nn.Sequential(
            nn.Linear(830 * 2, 768),
            nn.ReLU(),
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def build_inputs(self, chars):
        inputs = self.tokenizer(chars, return_tensors='pt', add_special_tokens=False)
        input_pinyins = self.encoder.input_helper.convert_tokens_to_pinyin_embeddings(inputs['input_ids'].view(-1))
        images = self.encoder.input_helper.convert_tokens_to_images(inputs['input_ids'].view(-1), None)
        return inputs.to("cuda"), input_pinyins.to("cuda"), images.to("cuda")

    def forward(self, inputs):
        with torch.no_grad():
            _, a_features = self.encoder(*self.build_inputs(inputs[0]), output_hidden_states=True)
            _, b_features = self.encoder(*self.build_inputs(inputs[1]), output_hidden_states=True)

        # a_features = self.encoder(self.build_inputs(inputs[0]))
        # b_features = self.encoder(self.build_inputs(inputs[1]))

        return self.cls(torch.concat([a_features.squeeze(), b_features.squeeze()], dim=-1).squeeze(0)).view(-1)


class GlyphPhoneticProbeTrain(object):

    def __init__(self):
        super(GlyphPhoneticProbeTrain, self).__init__()
        self.args = self.parse_args()

        def probe_collate_fn(batch):
            datas = [[], []]
            labels = []
            for data, label in batch:
                datas[0].append(data[0])
                datas[1].append(data[1])
                labels.append(label)
            return datas, torch.FloatTensor(labels)

        self.train_loader, self.valid_loader = create_dataloader(self.args, probe_collate_fn)

        self.model = GlyphPhoneticModel().to("cuda")
        self.model.train()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-4)

        self.total_step = 0
        self.current_epoch = 0

        self.criteria = nn.BCELoss()

    def compute_loss(self, outputs, targets):
        return self.criteria(outputs, targets)

    def train_epoch(self):
        self.model = self.model.train()
        progress = tqdm(self.train_loader, desc="Epoch {} Training".format(self.current_epoch))
        for i, (inputs, targets) in enumerate(progress):
            inputs, targets = inputs.to(self.args.device) if 'to' in dir(inputs) else inputs, \
                              targets.to(self.args.device) if 'to' in dir(targets) else targets

            self.optimizer.zero_grad()

            outputs = self.model(inputs)
            loss = self.compute_loss(outputs, targets)
            loss.backward()
            self.optimizer.step()

            self.total_step += 1

            accuracy = ((outputs >= 0.5) == targets.bool()).sum() / len(outputs)

            progress.set_postfix({
                'loss': loss.item(),
                'accuracy': accuracy.item()
            })

    def train(self):
        for epoch in range(self.current_epoch, self.args.epochs):
            try:
                self.train_epoch()
                self.validate()

                self.current_epoch += 1
            except BaseException as e:
                traceback.print_exc()
                exit()

        print("Finish Training. The best model is saved to", self.args.model_path)

    def validate(self):
        self.model = self.model.eval()

        total_correct_num = 0
        total_num = 0

        progress = tqdm(self.valid_loader, desc="Epoch {} Validation".format(self.current_epoch))
        for inputs, targets in progress:
            inputs, targets = inputs.to(self.args.device) if 'to' in dir(inputs) else inputs, \
                              targets.to(self.args.device) if 'to' in dir(targets) else targets

            outputs = self.model(inputs)
            total_correct_num += ((outputs >= 0.5) == targets.bool()).sum().item()
            total_num += len(outputs)

            progress.set_postfix({
                'accuracy': total_correct_num / total_num
            })

        accuracy = total_correct_num / total_num

        print("Accuracy: {}".format(accuracy))

    def parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--batch-size', type=int, default=32, help='The batch size of training.')
        parser.add_argument('--data-type', type=str, default="phonetic")
        parser.add_argument('--valid-ratio', type=float, default=0.2,
                            help='The ratio of splitting validation set.')
        parser.add_argument('--device', type=str, default='cuda',
                            help='The device for training. auto, cpu or cuda')
        parser.add_argument('--epochs', type=int, default=25, help='The number of training epochs.')

        args = parser.parse_known_args()[0]
        print(args)

        if args.device == 'auto':
            args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            args.device = torch.device(args.device)

        print("Device:", args.device)

        return args


if __name__ == '__main__':
    train = GlyphPhoneticProbeTrain()
    train.train()
