consonantList = ["b", "p", "m", "f", "d", "t", "n", "l", "g", "k", "h", "j", "q", "x", "zh", "ch", "sh", "r", "z", "c",
                 "s", "y", "w"]
vowelList = ["a", "o", "e", "i", "u", "v", "u:", "er", "ao", "ai", "ou", "ei", "ia", "iao", "iu", "iou", "ie", "ui",
             "uei", "ua", "uo", "uai", "u:e", "ve", "an", "en", "in", "un", "uen", "vn", "u:n", "ian", "uan", "u:an",
             "van", "ang", "eng", "ing", "ong", "iang", "iong", "uang", "ueng"]

sim_initials = ["b,p", "g,k", "h,f", "d,t", "n,l", "zh,z,j", "ch,c,q", "sh,s,x", "y,w"]
sim_finals = ["a,an,ang", "ia,ian,iang", "ua,uan,uang,u:an", "ao,iao", "ai,uai", "o,io,iou,iu,ou,uo", "ong,iong",
              "er,e", "u:e,ve,ue,ie,ei,uei,ui", "en,eng", "uen,un,ueng", "i,in,ing", "u,v,u:n,vn"]

sim_initials = [set(item.split(",")) for item in sim_initials]
sim_finals = [set(item.split(",")) for item in sim_finals]


class Pinyin:
    # Source Code: https://github.com/System-T/DimSim/blob/master/dimsim

    def __init__(self, pinyinstr:str):
        if pinyinstr[-1].isnumeric():
            self.tone = int(pinyinstr[-1])
            self.locp = pinyinstr[0:-1].lower()
        else:
            self.tone = 0
            self.locp = pinyinstr.lower()
        self.consonant, self.vowel = self.parseConsonant(self.locp)
        self.pinyinRewrite()

    def parseConsonant(self, pinyin):
        for consonant in consonantList:
            if pinyin.startswith(consonant):
                return (consonant, pinyin[len(consonant):])
        # it's a vowel without consonant
        if pinyin in vowelList:
            return None, pinyin.lower()

        print("Invalid Pinyin, please check!")
        return None, None

    def toStringNoTone(self):
        return "{}{}".format(self.consonant, self.vowel)

    def toStringWithTone(self):
        return "{}{}{}".format(self.consonant, self.vowel, self.tone)

    def toString(self):
        return "{}{}{}".format(self.consonant, self.vowel, self.tone)

    def pinyinRewrite(self):
        import re
        yVowels = {"u", "ue", "uan", "un", "u:", "u:e", "u:an", "u:n"}
        tconsonant = {"j", "g", "x"}
        if 'v' in self.vowel:
            self.vowel = self.vowel.replace("v", "u:")

        if self.consonant is None or self.consonant == "":
            self.consonant = ""
            return
        if self.consonant == "y":
            if self.vowel in yVowels:
                if "u:" not in self.vowel:
                    self.vowel = self.vowel.replace("u", "u:")
            else:
                self.vowel = "i" + self.vowel
                regex = re.compile("i+")
                self.vowel = self.vowel.replace("iii", "i")
                self.vowel = self.vowel.replace("ii", "i")
            self.consonant = "y"

        if self.consonant == "w":
            self.vowel = "u" + self.vowel
            self.vowel = self.vowel.replace("uuu", "u")
            self.vowel = self.vowel.replace("uu", "u")
            self.consonant = "w"

        if (self.consonant in tconsonant) and (self.vowel == "u") or (self.vowel == "v"):
            self.vowel = "u:"

        if self.vowel == "iou":
            self.vowel = "iu"

        if self.vowel == "uei":
            self.vowel = "ui"

        if self.vowel == "uen":
            self.vowel = "un"


def pinyin_is_sim(pinyin1, pinyin2):
    pinyin1 = Pinyin(pinyin1)
    pinyin2 = Pinyin(pinyin2)
    initial_pair = {pinyin1.consonant, pinyin2.consonant}
    initial_sim = False
    for item in sim_initials:
        if initial_pair.issubset(item):
            initial_sim = True
            break

    final_pair = {pinyin1.vowel, pinyin2.vowel}
    final_sim = False
    for item in sim_finals:
        if final_pair.issubset(item):
            final_sim = True
            break

    return initial_sim and final_sim


if __name__ == '__main__':
    print(pinyin_is_sim("nv", "nu"))