var vo={none:3,flow:1,orb:2,drift:4,swell:5,caustics:6,contour:7,aurora:8,ripple:9,halo:10,interference:11,mesh:12,cascade:13,constellation:14},Ir=e=>e?`
const float VSCALE=6.;
vec4 decodeFluid(vec4 t){return vec4((t.xy*2.-1.)*VSCALE,t.z,t.w);}
vec4 encodeFluid(vec4 v){return vec4(clamp(v.xy/VSCALE*.5+.5,0.,1.),clamp(v.z,0.,1.),1.);}`:`
vec4 decodeFluid(vec4 t){return t;}
vec4 encodeFluid(vec4 v){return vec4(v.xyz,1.);}`,Br=e=>`#version 300 es
precision highp float;
precision highp sampler2D;
uniform sampler2D uSource, uFluid;
uniform vec2 uResolution, uCell;
uniform float uCount, uBlend;
const float BAYER[16]=float[16](0.,8.,2.,10.,12.,4.,14.,6.,3.,11.,1.,9.,15.,7.,13.,5.);
uniform vec4 uSourceMap;
uniform int uScene;
uniform vec3 uInk, uHighlight;
uniform vec4 uBackground;
uniform float uBrightness, uContrast, uGamma, uInvert, uVignette, uSourceColor;
uniform float uRefraction, uGlow, uTime, uDrift, uFluidOn, uLod, uSaturation, uBackdrop, uShade, uHighlightAuto, uFluidMode;
uniform vec2 uFluidTexel;
uniform int uHoverMode;
uniform vec3 uLight;
uniform vec3 uTrail[12];
uniform float uLightRadius, uLightGain, uClock, uFlicker, uTrailAmount;
uniform sampler2D uTrailMap, uScope;
uniform float uScopeMode;
uniform vec2 uLevels;
uniform int uMark;
uniform float uScale, uClearFeather, uClearShape, uOrder;
uniform vec4 uClear;
uniform vec4 uPulse[4];
layout(location=0) out vec4 outInk;
layout(location=1) out vec4 outUnder;
${Ir(e)}
float hash(vec2 p){p=fract(p*vec2(123.34,456.21));p+=dot(p,p+45.32);return fract(p.x*p.y);}
float noise(vec2 p){vec2 i=floor(p),f=fract(p),u=f*f*(3.-2.*f);
  return mix(mix(hash(i),hash(i+vec2(1,0)),u.x),mix(hash(i+vec2(0,1)),hash(i+vec2(1,1)),u.x),u.y);}
float fbm(vec2 p){float v=0.,a=.5;for(int i=0;i<5;i++){v+=a*noise(p);p=p*2.03+vec2(1.7,9.2);a*=.5;}return v;}
vec3 generated(vec2 uv,float aspect){
  vec2 p=(uv-.5)*vec2(aspect,1.)*uScale;
  if(uScene==1){
    float t=uTime*.08;
    vec2 q=vec2(fbm(p*1.6+t),fbm(p*1.6-t+3.1));
    return vec3(smoothstep(.28,.86,fbm(p*1.8+q*1.6+vec2(t*.6,-t*.4))));
  }
  if(uScene==2){
    float r=.36,d=length(p),a=uTime*.25;
    vec3 l=normalize(vec3(cos(a)*.8,.55,sin(a)*.6+.6));
    if(d<r){
      vec3 n=normalize(vec3(p.x,-p.y,sqrt(r*r-d*d)));
      float diffuse=max(dot(n,l),0.),rim=pow(1.-n.z,2.5);
      float lon=atan(n.x,n.z)+uTime*.35,lat=asin(clamp(n.y,-1.,1.));
      float bands=.5+.25*sin(lat*14.)+.25*sin(lon*6.);
      return vec3(clamp(diffuse*.85+rim*.5+bands*.12,0.,1.));
    }
    return vec3(exp(-pow((d-r)*9.,2.))*.35);
  }
  if(uScene==4){
    // Drift: a few broad lobes that wander slowly. Built for halftone, where large soft shapes read best.
    float t=uTime*.045;
    vec2 q=p*1.15;
    vec2 w=vec2(fbm(q*.7+vec2(t,-t*.7)),fbm(q*.7+vec2(4.3-t*.8,t+1.7)));
    float n=fbm(q+w*1.3+vec2(t*.4,t*.25));
    return vec3(smoothstep(.36,.74,n));
  }
  if(uScene==5){
    // Swell: three crossing swells, the second bent by the first, so crests roll diagonally instead of pooling.
    float t=uTime;
    float s=sin((dot(p,vec2(.94,.33))/.95-t/9.)*6.2831853);
    float c=sin((dot(p,vec2(.6,-.8))/.58-t/13.)*6.2831853+s*.9);
    float k=sin((dot(p,vec2(-.29,.96))/.36+t/17.)*6.2831853);
    return vec3(smoothstep(.3,.9,.5+.5*(s*.5+c*.35+k*.15)));
  }
  // The kit's background scenes. Each starts a little way in, so a reduced-motion still is already composed.
  float t=uTime+40.,px=uCell.y/uResolution.y*uScale;
  if(uScene==6){
    // Caustics: light focused by a moving water surface. Two layers of warped sine waves drift against each other; each draws
    // smooth filaments along its zero line, about a character wide. Where they cross the light lifts a little, never to a blot.
    float tt=t*.12,a=0.,b=0.;
    for(int l=0;l<2;l++){
      float f=float(l),sc=5.4+f*2.4;
      vec2 q=(l==0?p:mat2(.8,-.6,.6,.8)*p)*sc;
      q+=vec2(sin(q.y*.45+tt*(1.+f*.3)),cos(q.x*.4-tt*(.8+f*.4)))*1.3;
      float v=sin(q.x+1.4*sin(q.y*.7+tt))+sin(q.y*1.1+1.2*sin(q.x*.8-tt*1.3))*.9;
      float wv=max(fwidth(v),1e-4);
      float line=(1.-smoothstep(.2,1.5,abs(v)/wv))*(l==0?.78:.62)+exp(-abs(v)/(wv*5.))*.16;
      if(l==0)a=line;else b=line;
    }
    float pulse=.85+.15*sin(tt*3.+p.x*1.7-p.y*1.1);
    float light=max(a,b)+.3*min(a,b);
    return vec3(clamp(light*light*pulse,0.,.78));
  }
  if(uScene==7){
    // Contour: a slow terrain read as survey lines about a character wide at any size; every fifth is an index line.
    float tt=t*.02;
    vec2 q=p*1.15+vec2(tt,-tt*.6);
    float h=noise(q)*.62+noise(q*2.07+3.1)*.27+noise(q*4.3-tt+7.)*.11;
    h+=.08*sin(tt*2.+p.x*.8);
    float k=h*13.,w=max(fwidth(k),1e-4),d=abs(fract(k+.5)-.5)/w;
    float major=mod(floor(k+.5),5.)<.5?1.:0.;
    return vec3((1.-smoothstep(.35,1.25-major*.25,d))*mix(.55,1.,major));
  }
  if(uScene==8){
    // Aurora: three curtains at different distances hang from the top. Far ones sit higher and dimmer; the nearest is tallest
    // and brightest. Each folds slowly and is brightest along its lower hem, fading upward. The rays are set per character
    // column, so the curtains read as vertical strokes at any stage size, from a small card to a phone.
    float tt=t*.05,a=0.,cw=uCell.x/uResolution.x;
    float col=floor(uv.x/cw+.5);
    for(int n=0;n<3;n++){
      float f=float(n),near=f/2.;
      float x=p.x*(1.3-near*.4)+f*3.7;
      float hem=-.28+near*.3+.16*sin(x*1.3+tt*(.9+f*.25)+f*2.)+.045*sin(x*3.4-tt*1.5+f);
      float d=hem-p.y;
      float body=d>0.?exp(-d/(.14+near*.26))*(.45+.55*exp(-d/.03)):exp(-d*d/.0002);
      float fold=.5+.5*sin(x*2.1+tt*.9+f*1.7+.7*sin(x*.8-tt*.6));
      float ray=mix(hash(vec2(col,f*7.1)),noise(vec2(col*.18+f*9.,tt*.7+f)),.55);
      float layer=clamp(body*(.1+.9*fold*fold)*(.2+.8*ray)*(.4+near*.6)*4.2,0.,1.);
      a=1.-(1.-a)*(1.-layer);
    }
    return vec3(smoothstep(.12,.9,a));
  }
  if(uScene==9){
    // Ripple: still water. Drops land here and there, each a few rings that widen and fade; the pointer drops its own.
    float tone=0.,g=.72;
    vec2 q=p/g,id=floor(q);
    for(int j=-1;j<=1;j++)for(int i=-1;i<=1;i++){
      vec2 c=id+vec2(i,j),h=vec2(hash(c),hash(c+7.3));
      float period=5.+h.x*5.,age=fract(t*.3/period*4.+h.y);
      vec2 at=(c+.2+.6*h)*g;
      float d=length(p-at),r=age*.62;
      float rings=max(0.,cos((d-r)*34.))*exp(-pow((d-r)/.11,2.));
      tone+=rings*pow(1.-age,1.8)*smoothstep(0.,.06,age)*(.7+.3*h.x);
    }
    return vec3(clamp(tone*1.25,0.,1.));
  }
  if(uScene==10){
    // Halo: a slow breathing corona that frames the content, its rays turning. The ring fits the stage: wide on a desktop
    // hero, tall on a phone, so the whole halo stays on screen; the middle stays empty for a headline.
    vec2 r=vec2(min(.46,aspect*.4),min(.34,.36/max(aspect,.01)*.5+.2));
    vec2 e=p/r;
    float d=length(e),a=atan(e.y,e.x),tt=t*.07,k=min(r.x,r.y);
    float breathe=1.+.04*sin(tt*3.1);
    float rays=.5+.5*noise(vec2(a*3.2+tt,tt*.5))*(.6+.4*noise(vec2(a*7.-tt*1.2,3.+tt*.4)));
    float s=(d-breathe)*k;
    float rim=exp(-pow(s/.03,2.));
    float corona=exp(-max(0.,s)/(.1+.08*rays))*smoothstep(-.04,.015,s)*rays*smoothstep(.36,.12,s);
    return vec3(clamp(rim*(.55+.45*rays)+corona*.85,0.,1.));
  }
  if(uScene==11){
    // Interference: two ring grids from slowly orbiting centres. Where they cross, broad moire bands open and close.
    float tt=t*.05;
    vec2 a=vec2(-.3+.12*sin(tt),.07*cos(tt*1.3)),b=vec2(.3+.1*cos(tt*.8),-.06*sin(tt*1.1));
    float f=6.2831853/max(px*4.6,.004);
    float m=cos(length(p-a)*f)+cos(length(p-b)*f);
    return vec3(smoothstep(.7,2.,m));
  }
  if(uScene==12){
    // Mesh: four soft lights on slow Lissajous paths, a gradient mesh written in characters.
    float tt=t*.07,m=0.;
    for(int n=0;n<4;n++){
      float f=float(n)*1.7;
      vec2 c=vec2(.66*sin(tt*(.7+f*.13)+f),.34*cos(tt*(.9-f*.07)+f*1.3));
      float s=.3+.07*sin(tt+f);
      m+=exp(-dot(p-c,p-c)/(s*s))*(.8-float(n)*.1);
    }
    return vec3(smoothstep(.06,1.1,m));
  }
  if(uScene==13){
    // Cascade: streaks fall in three depths. Far ones are frequent, faint, short and slow; near ones rare, long, bright and
    // quick. Each has a bright head and a tail that fades behind it, and a column can carry two streaks at once.
    float tone=0.,cw=uCell.x/uResolution.x,ch=uCell.y/uResolution.y;
    float col=floor(uv.x/cw+.5);
    for(int n=0;n<3;n++){
      float f=float(n);
      for(int k=0;k<2;k++){
        float seed=f*13.1+float(k)*41.7,h=hash(vec2(col,seed));
        if(h<1.-(.5-f*.15))continue;
        float speed=(.045+f*.07)*(.75+.5*hash(vec2(col,seed+2.)));
        float tail=.12+f*.16,y=fract(t*speed+hash(vec2(col,seed+5.)))*(1.+tail*2.)-tail;
        float d=y-uv.y;
        if(d<0.||d>tail)continue;
        float head=d<ch*1.2?1.:0.;
        tone=max(tone,max(head*(.55+f*.22),pow(1.-d/tail,1.7)*(.28+f*.3)));
      }
    }
    return vec3(clamp(tone,0.,1.));
  }
  if(uScene==14){
    // Constellation: points wander in their own cells; neighbours close enough are linked by fine lines. Each point is a
    // bright core in a soft halo, so it reads at card size as well as full width.
    float g=.23,tt=t*.18;
    vec2 q=p/g,id=floor(q);
    vec2 pts[9];
    for(int j=-1;j<=1;j++)for(int i=-1;i<=1;i++){
      vec2 c=id+vec2(i,j);
      float h=hash(c),k=hash(c+3.7);
      pts[(j+1)*3+i+1]=c+.5+.36*vec2(sin(tt*(.6+h)+h*6.28),cos(tt*(.5+k)+k*6.28));
    }
    float line=0.,star=0.,lw=max(px/g,.02);
    for(int n=0;n<9;n++){
      vec2 e=q-pts[n];
      float d2=dot(e,e),tw=.8+.2*sin(tt*3.+float(n)*1.7);
      star=max(star,max(exp(-d2/(lw*lw*1.6)),.38*exp(-d2/(lw*lw*7.)))*tw);
      for(int m=n+1;m<9;m++){
        vec2 a=pts[n],ba=pts[m]-a;
        float len=length(ba);
        if(len>1.35)continue;
        float hh=clamp(dot(q-a,ba)/dot(ba,ba),0.,1.);
        float d=length(q-a-ba*hh);
        line=max(line,(1.-smoothstep(lw*.3,lw*.75,d))*smoothstep(1.35,.6,len)*.72);
      }
    }
    return vec3(clamp(max(line,star),0.,1.));
  }
  return vec3(0.);
}
// Hover scope: 1 where the pointer response may act. The map holds the source's soft subject coverage in alpha.
float hoverScope(vec2 uv){
  if(uScopeMode<.5||uScene!=0)return 1.;
  vec2 s=uv*uSourceMap.xy+uSourceMap.zw;
  float c=(s.x<0.||s.y<0.||s.x>1.||s.y>1.)?0.:texture(uScope,s).a;
  return uScopeMode<1.5?c:1.-c;
}
void main(){
  vec2 cell=floor(gl_FragCoord.xy);
  vec2 uv=(cell+.5)*uCell/uResolution;
  float aspect=uResolution.x/uResolution.y;
  float hs=hoverScope(uv);
  vec4 fluid=vec4(0.);
  // lift: signed tone change. The solver's flow brightens one side of each eddy and opens the other into gaps.
  float lift=0.,wake=0.,bend=1.,swirl=0.,speed=0.;
  if(uFluidOn>.5){
    vec4 raw=texture(uFluid,vec2(uv.x,1.-uv.y));
    if(uFluidMode>.5){
      vec2 v=raw.xy*uFluidTexel*vec2(aspect,1.)*hs;
      fluid=vec4(v,0.,1.);
      swirl=clamp(dot(v,vec2(.4,.65))*uGlow*.32,-.65,.65);
      speed=clamp(length(v)*uGlow*.055,0.,.38);
      bend=.4;
    } else {
      fluid=decodeFluid(raw);
      fluid.xyz*=hs;
      wake=clamp(fluid.z*uGlow,0.,1.);
      lift=wake*.55;
    }
  }
  vec2 drift=uDrift*.006*vec2(sin(uv.y*5.3+uTime*.6),cos(uv.x*4.1-uTime*.5));
  vec2 sampleAt=uv-vec2(fluid.x/aspect,-fluid.y)*uRefraction*bend+drift;
  float spot=0.;
  if(uHoverMode==4||uHoverMode==5){
    vec2 d=(uv-uLight.xy)*vec2(aspect,1.);
    float q2=dot(d,d)/(uLightRadius*uLightRadius*2.2);
    // Loupe: flat-topped zoom with a soft rim. Reveal: a soft spot in the source's colours.
    if(uHoverMode==4) sampleAt-=(uv-uLight.xy)*exp(-q2*q2)*.45*uLight.z*min(1.,uLightGain*1.6)*hs;
    else spot=exp(-q2*1.6)*uLight.z*min(1.,uLightGain*1.6)*hs;
  }
  vec3 rgb,glow=vec3(0.);
  if(uScene==0){
    vec2 s=sampleAt*uSourceMap.xy+uSourceMap.zw;
    bool inside=!(s.x<0.||s.y<0.||s.x>1.||s.y>1.);
    rgb=inside?textureLod(uSource,s,uLod).rgb:vec3(0.);
    if(uBackdrop>0.){
      vec2 b=(uv-vec2(fluid.x/aspect,-fluid.y)*uRefraction*bend*.5)*uSourceMap.xy+uSourceMap.zw;
      glow=(b.x<0.||b.y<0.||b.x>1.||b.y>1.)?vec3(0.):textureLod(uSource,b,uLod+3.).rgb*uBackdrop;
    }
  } else rgb=generated(sampleAt,aspect);
  float tone=dot(rgb,vec3(.2126,.7152,.0722));
  tone=clamp((tone-uLevels.x)/max(uLevels.y-uLevels.x,.01),0.,1.);
  tone=clamp((tone-.5)*uContrast+.5+uBrightness,0.,1.);
  tone=pow(tone,1./uGamma);
  if(uInvert>.5)tone=1.-tone;
  float flick=0.,trailFold=0.;
  // Reveal: under the spot the characters take the source's own colour and a little more ink.
  if(spot>0.)lift=spot*.22;
  if(uHoverMode==2){
    // Lantern: a soft light and its fading tail lift the characters they pass. Nothing moves the image.
    vec2 ap=vec2(aspect,1.),d=(uv-uLight.xy)*ap;
    float r2=uLightRadius*uLightRadius,glowL=uLight.z*exp(-dot(d,d)/r2);
    for(int i=0;i<12;i++){vec2 e=(uv-uTrail[i].xy)*ap;glowL+=uTrail[i].z*exp(-dot(e,e)/(r2*.55));}
    glowL=min(glowL,1.)*hs;
    float grain=fract(sin(dot(cell,vec2(12.9898,78.233)))*43758.5453);
    lift=glowL*uLightGain*(.7+.6*grain);
    wake=glowL*.4;
    // A few letters at the light's rim step up the ramp and back, as if being retyped.
    flick=uFlicker*step(.9,fract(grain*13.7+floor(uClock*7.+grain*5.)*.37))*glowL*(1.-glowL)*4.;
  }
  if(uHoverMode==3){
    // Trail: the studio's ink field, as a 16-bit density. Sparse marks fill in, dense marks open up.
    vec4 ink=texture(uTrailMap,uv);
    float density=(ink.r*65280.+ink.g*255.)/65535.*uTrailAmount;
    float t=clamp(density/.38,0.,1.);
    trailFold=t*t*(3.-2.*t)*hs;
  }
  if(uFluidMode>.5&&uHoverMode==1){
    // Dark ground: any motion lights the characters. Lit ground: the flow lights one side of each eddy and opens the other.
    // A little cell-to-cell grain keeps the lit area a shimmer of letters rather than a solid block.
    float grain=fract(sin(dot(cell,vec2(12.9898,78.233)))*43758.5453);
    lift=mix(abs(swirl)*.5+speed,swirl+speed*.35,smoothstep(.12,.55,tone))*(.65+.7*grain);
    wake=clamp(lift*1.6,0.,1.);
  }
  tone=clamp(tone+lift,0.,1.);
  tone+=(1.-2.*tone)*trailFold;
  // Pulses: rings sent by the interface (field.pulse()), e.g. on a save or a notification.
  for(int i=0;i<4;i++){
    if(uPulse[i].w<=0.)continue;
    float dist=length((uv-uPulse[i].xy)*vec2(aspect,1.)),front=uPulse[i].z*.95;
    float ring=exp(-pow((dist-front)/.1,2.))*uPulse[i].w*exp(-uPulse[i].z*.85);
    tone=clamp(tone+ring*.85,0.,1.);
  }
  tone*=1.-uVignette*smoothstep(.35,1.05,length((uv-.5)*vec2(min(aspect,1.8),1.)*2.));
  if(uClear.z>0.){
    // Clear zone: the field fades out around content so text above it stays readable.
    vec2 half_=uClear.zw*.5*vec2(aspect,1.),q=(uv-uClear.xy-uClear.zw*.5)*vec2(aspect,1.);
    float dz;
    if(uClearShape>.5){vec2 e=abs(q)-half_;dz=length(max(e,0.))+min(max(e.x,e.y),0.);}
    else dz=(length(q/max(half_,vec2(1e-4)))-1.)*min(half_.x,half_.y);
    tone*=smoothstep(0.,uClearFeather,dz);
  }
  // Blend: a shade between two characters takes the upper one in the share of cells a 4x4 Bayer pattern gives it.
  float level=tone*(uCount-1.),upper=fract(level)>=.5?1.:0.;
  if(uBlend>.5&&tone>.02&&tone<.98){
    ivec2 q=ivec2(mod(gl_FragCoord.xy,4.));
    upper=fract(level)>(BAYER[q.y*4+q.x]+.5)/16.?1.:0.;
  }
  float index=min(uCount-1.,floor(level)+upper+(flick>.5&&tone>.02?1.:0.));
  if(uOrder>.5){
    // Text order: the charset after its blank is read in sequence along each row, each row a few letters on, so words stay
    // legible. Every letter shows from a tone of .6; below it a 4x4 Bayer screen thins them out toward the field's quiet edges.
    ivec2 q=ivec2(mod(cell,4.));
    float n=max(1.,uCount-1.);
    index=tone>.04+(BAYER[q.y*4+q.x]+.5)/16.*.56?1.+mod(cell.x+cell.y*(floor(n*.38)+1.),n):0.;
  }
  vec3 hue=rgb/max(max(rgb.r,rgb.g),max(rgb.b,.03));
  hue=mix(vec3(dot(hue,vec3(.2126,.7152,.0722))),hue,uSaturation);
  vec3 ink=mix(uInk*mix(1.,.3+.7*tone,uShade),clamp(hue*(.38+.62*tone),0.,1.),uSourceColor);
  vec3 lifted=uHighlightAuto>.5?ink/max(max(ink.r,ink.g),max(ink.b,.06)):uHighlight;
  ink=mix(ink,lifted,wake*.7);
  if(spot>0.)ink=mix(ink,clamp(rgb/max(max(rgb.r,rgb.g),max(rgb.b,.08))*(.45+.55*tone),0.,1.),spot);
  float underAlpha=max(uBackground.a,min(1.,max(glow.r,max(glow.g,glow.b))));
  vec3 under=min(uBackground.rgb*uBackground.a+glow,vec3(underAlpha));
  // Glyphs keep their atlas index, exact in 8 bits; the other marks keep the tone, and empty where the charset is blank.
  outInk=vec4(ink,uMark==0?index/255.:uMark==1?(tone>.003?tone:0.):index>0.?max(tone,1./255.):0.);
  outUnder=vec4(under,underAlpha);
}`,qr=`#version 300 es
precision highp float;
precision highp sampler2D;
uniform sampler2D uAtlas, uCells, uUnder;
uniform vec2 uResolution, uCell, uPitch;
uniform float uColumns, uDotScale, uSourceColor;
uniform vec2 uDir;
uniform int uMark;
out vec4 outColor;
const float BAYER[16]=float[16](0.,8.,2.,10.,12.,4.,14.,6.,3.,11.,1.,9.,15.,7.,13.,5.);
// The Studio's print defaults: roughness .35, seed 7, registration .2 and its coral second ink. The screen angle is uDir (-35deg).
const float ROUGH=.35,SEED=7.;
const vec3 SECOND=vec3(.925,.404,.322);
float hash(vec2 p){p=fract(p*vec2(123.34,456.21));p+=dot(p,p+45.32);return fract(p.x*p.y);}
// Coverage of a signed distance given in device pixels.
float cover(float d){return clamp(.5-d,0.,1.);}
float box(vec2 p,vec2 lo,vec2 hi){vec2 a=clamp(p-lo,0.,1.)*clamp(hi-p,0.,1.);return a.x*a.y;}
float disc(vec2 p,vec2 c,float r){return r>.05?cover(length(p-c)-r):0.;}
float ellipse(vec2 p,vec2 c,vec2 r,float a){vec2 d=p-c;d=vec2(cos(a)*d.x+sin(a)*d.y,-sin(a)*d.x+cos(a)*d.y);float k=length(d/r);return cover((k-1.)*min(r.x,r.y));}
// World-anchored parallel cuts across direction d: period and width in cell heights, so lines run on from cell to cell.
float cuts(vec2 w,vec2 d,float period,float width,float px){
  if(width<=0.)return 0.;
  float proj=dot(vec2(-d.y,d.x),w);
  return cover((abs(fract(proj/period+.5)-.5)*period-width*.5)*px);
}
// Lays an ink over what is already down, multiplying where they overlap, as transparent inks do on paper.
void lay(inout vec3 rgb,inout float a,vec3 c,float k){
  float next=a+k*(1.-a);
  rgb=next>0.?(rgb*a*(1.-k)+mix(c,rgb*c,a)*k)/next:c;
  a=next;
}
void main(){
  vec2 DIR=uDir;
  vec2 frag=vec2(gl_FragCoord.x,uResolution.y-gl_FragCoord.y);
  // The pitch is the atlas cell, or a fractional one under cellFit 'fill', where each glyph sits centred in its stretched cell.
  vec2 cell=floor(frag/uPitch);
  vec4 ink=texelFetch(uCells,ivec2(cell),0),under=texelFetch(uUnder,ivec2(cell),0);
  vec2 local=frag-cell*uPitch;
  float coverage;
  vec3 rgb=ink.rgb;
  if(uMark==0){
    float index=floor(ink.a*255.+.5);
    ivec2 tile=ivec2(int(mod(index,uColumns)),int(floor(index/uColumns)));
    vec2 texel=floor(local-floor((uPitch-uCell)*.5));
    coverage=all(greaterThanEqual(texel,vec2(0.)))&&all(lessThan(texel,uCell))?texelFetch(uAtlas,tile*ivec2(uCell)+ivec2(texel),0).a:0.;
  } else if(uMark==1){
    // Halftone: dot area follows tone, so midtones read true; edges are antialiased to one device pixel.
    float r=sqrt(ink.a)*min(uPitch.x,uPitch.y)*.5*uDotScale;
    coverage=ink.a>.003?clamp(r-length(local-uPitch*.5)+.5,0.,1.):0.;
  } else if(ink.a<=0.){
    coverage=0.;
  } else {
    float tone=ink.a,side=min(uPitch.x,uPitch.y);
    vec2 q=local/uPitch,c=uPitch*.5,p=local+.5;
    // Tiles fade with the tone in one ink, as the Studio's do; in source colour the colour already carries it.
    float fade=uSourceColor>.5?1.:tone,gap=max(.5,uPitch.x*.06),grain=hash(cell+SEED);
    vec2 w=(cell+q)*uPitch/uPitch.y;
    if(uMark==2)coverage=fade;
    else if(uMark==3)coverage=box(p,vec2(0.),uPitch-gap)*fade;
    else if(uMark==4){
      // Lego: a gapped brick, lit from above, with a raised stud ringed by its shadow.
      coverage=box(p,vec2(0.),uPitch-gap)*fade;
      rgb*=1.06-.14*q.y;
      float r=side*.28,d=length(local-c)-r,lit=clamp(dot(c-local,vec2(.6,.8))/r,-1.,1.);
      rgb=mix(rgb,mix(rgb,vec3(1.),.2+.12*lit),cover(d));
      rgb=mix(rgb,rgb*.72,cover(abs(d)-max(.5,side*.04)));
    } else if(uMark==5){
      // Voxel: an isometric cube; top, left and right faces at full, .72 and .42 strength.
      float h=.5*abs(q.x-.5);
      coverage=cover(-min(q.y-h,1.-h-q.y)*uPitch.y)*cover(-min(q.x,1.-q.x)*uPitch.x);
      coverage*=mix(q.x<.5?.72:.42,1.,cover((q.y-(.5-h))*uPitch.y))*fade;
    } else if(uMark==6){
      // Disco: a gapped mirror tile, its upper facet catching light and its lower in shade.
      coverage=box(p,vec2(0.),uPitch-gap)*fade;
      rgb=q.y<.35?mix(rgb,vec3(1.),.1+grain*.25):q.y>.65?rgb*.75:rgb;
    } else if(uMark==7){
      // Dither: an ordered 4x4 Bayer screen at about five dots a cell, lit where the cell's tone passes the threshold.
      ivec2 b=ivec2(mod(floor(frag/max(1.,floor(side/5.))),4.));
      coverage=tone>(BAYER[b.y*4+b.x]+.5)/16.?1.:0.;
    } else if(uMark==8){
      // Hex: a flat-topped hexagon filling the cell.
      vec2 h=abs((local-c)/(uPitch*.49));
      coverage=cover(max(h.y-.866,h.x*.866+h.y*.5-.866)*side*.49)*fade;
    } else if(uMark==9){
      // LED: a round lamp with a soft halo and a specular glint, separated by an unlit matrix.
      float r=side*.39,d=length(local-c);
      coverage=max(cover(d-r),.3*exp(-pow(max(0.,d-r)/(side*.12),2.)))*fade;
      rgb=mix(rgb,vec3(1.),.35*disc(local,uPitch*vec2(.43,.4),side*.11));
    } else if(uMark==10){
      // CMYK: the cell's hue separated into offset cyan, magenta and yellow dots, each sized by its share of the tone. The
      // key screen carries the neutral share in the ink's own lightness, since a light ink on a dark page has no black.
      float top=max(rgb.r,max(rgb.g,rgb.b)),low=min(rgb.r,min(rgb.g,rgb.b));
      vec3 cmy=(top-rgb)/max(top,.001);
      float a=0.;vec3 col=vec3(1.);
      lay(col,a,vec3(top),disc(local,c,side*.48*sqrt(tone*low/max(top,.001))));
      lay(col,a,vec3(0.,1.,1.),disc(local,uPitch*vec2(.36,.36),side*.48*sqrt(tone*cmy.x)));
      lay(col,a,vec3(1.,0.,1.),disc(local,uPitch*vec2(.64,.36),side*.48*sqrt(tone*cmy.y)));
      lay(col,a,vec3(1.,1.,0.),disc(local,uPitch*vec2(.5,.66),side*.48*sqrt(tone*cmy.z)));
      rgb=col;coverage=a;
    } else if(uMark==11||uMark==16){
      // Stipple and pointillism: four jittered marks a cell, their size following the tone.
      coverage=0.;vec3 col=rgb;
      for(int m=0;m<4;m++){
        vec2 o=vec2(float(m%2),float(m/2));
        float j=hash(cell*2.+o+SEED),jy=hash(cell+o*11.+SEED+1.);
        vec2 at=((o+.5)/2.+(vec2(j,jy)-.5)*ROUGH*.28)*uPitch;
        float r=min(min(min(at.x,uPitch.x-at.x),min(at.y,uPitch.y-at.y)),side*min(.245,.21*sqrt(tone)*(1.-ROUGH*j*.35)));
        float k=uMark==11?disc(local,at,r):ellipse(local,at,vec2(r,r*.64),j*3.14159);
        if(uMark==16&&k>coverage)col=uSourceColor>.5?clamp(rgb+(j-.5)*ROUGH*.25*vec3(1.,-.4,-1.),0.,1.):rgb*(1.+(j-.5)*ROUGH*.5);
        coverage=max(coverage,k);
      }
      rgb=col;
    } else if(uMark==12||uMark==13){
      // Engraving and crosshatch: parallel cuts whose weight follows the tone; crossings join only in the darker tones.
      float cv=tone*(1.-ROUGH*grain*.3),px=uPitch.y;
      coverage=cuts(w,DIR,.5,min(.47,.36*cv),px);
      if(uMark==13){
        coverage=max(coverage,cuts(w,vec2(-DIR.y,DIR.x),.5,min(.47,.36*max(0.,cv*1.5-.3)),px)*step(.2,cv));
        coverage=max(coverage,cuts(w,normalize(DIR+vec2(-DIR.y,DIR.x)),.5,min(.47,.36*max(0.,cv*2.-1.2)),px)*step(.6,cv));
      }
    } else if(uMark==14){
      // Woodblock: broad carved bands in five stepped weights, their edges wavering along the grain.
      float cv=ceil(tone*(1.-ROUGH*grain*.3)*5.)/5.,width=min(.94,.72*cv);
      float proj=dot(vec2(-DIR.y,DIR.x),w),line=floor(proj+.5),along=dot(DIR,w);
      float cut=sin(along*5.3+line*1.9+SEED)*.5+sin(along*13.1+line)*.25;
      coverage=cover((abs(proj-line)-width*.5*(1.+ROUGH*cut*.5))*uPitch.y);
    } else if(uMark==15){
      // Risograph: two offset screens, the main ink and the coral second ink, misregistered along the screen angle.
      float split=uSourceColor>.5?clamp(.5+(rgb.r-rgb.b)*.5,0.,1.):.5,r=side*min(.48,.44*sqrt(tone));
      float a=0.;vec3 col=rgb;
      lay(col,a,rgb,disc(local,c,r*sqrt(.35+(1.-split)*.65)));
      lay(col,a,SECOND,disc(local,c+DIR*.2*.38*side,r*sqrt(.3+split*.7)*(1.-ROUGH*hash(cell+SEED+13.)*.3)));
      rgb=col;coverage=a;
    } else coverage=0.;
  }
  outColor=vec4(rgb*coverage+under.rgb*(1.-coverage),coverage+under.a*(1.-coverage));
}`,Ie=`#version 300 es
precision highp float;
precision highp sampler2D;
uniform vec2 uTexel;
out vec4 outColor;
vec2 here(){return gl_FragCoord.xy*uTexel;}
`,xo=Ie+`
uniform sampler2D uVelocity;
uniform vec2 uFrom, uTo, uForce;
uniform float uAspect, uRadius;
float segment(vec2 p,vec2 a,vec2 b){vec2 pa=p-a,ba=b-a;float h=clamp(dot(pa,ba)/max(dot(ba,ba),1e-8),0.,1.);return length(pa-ba*h);}
void main(){
  vec2 uv=here(),scale=vec2(uAspect,1.);
  float d=segment(uv*scale,uFrom*scale,uTo*scale);
  outColor=vec4(texture(uVelocity,uv).xy+uForce*exp(-d*d/(uRadius*uRadius)),0.,1.);
}`,yo=Ie+`
uniform sampler2D uVelocity;
void main(){
  vec2 uv=here();
  float l=texture(uVelocity,uv-vec2(uTexel.x,0.)).y,r=texture(uVelocity,uv+vec2(uTexel.x,0.)).y;
  float t=texture(uVelocity,uv+vec2(0.,uTexel.y)).x,b=texture(uVelocity,uv-vec2(0.,uTexel.y)).x;
  outColor=vec4(.5*(r-l-t+b),0.,0.,1.);
}`,wo=Ie+`
uniform sampler2D uVelocity, uCurl;
uniform float uStrength, uDt;
void main(){
  vec2 uv=here();
  float l=texture(uCurl,uv-vec2(uTexel.x,0.)).x,r=texture(uCurl,uv+vec2(uTexel.x,0.)).x;
  float t=texture(uCurl,uv+vec2(0.,uTexel.y)).x,b=texture(uCurl,uv-vec2(0.,uTexel.y)).x,c=texture(uCurl,uv).x;
  vec2 force=.5*vec2(abs(t)-abs(b),abs(r)-abs(l));
  force=force/(length(force)+1e-4)*uStrength*c;
  force.y=-force.y;
  outColor=vec4(clamp(texture(uVelocity,uv).xy+force*uDt,-1000.,1000.),0.,1.);
}`,bo=Ie+`
uniform sampler2D uVelocity;
void main(){
  vec2 uv=here(),c=texture(uVelocity,uv).xy;
  float l=texture(uVelocity,uv-vec2(uTexel.x,0.)).x,r=texture(uVelocity,uv+vec2(uTexel.x,0.)).x;
  float t=texture(uVelocity,uv+vec2(0.,uTexel.y)).y,b=texture(uVelocity,uv-vec2(0.,uTexel.y)).y;
  if(uv.x-uTexel.x<0.)l=-c.x;
  if(uv.x+uTexel.x>1.)r=-c.x;
  if(uv.y+uTexel.y>1.)t=-c.y;
  if(uv.y-uTexel.y<0.)b=-c.y;
  outColor=vec4(.5*(r-l+t-b),0.,0.,1.);
}`,Eo=Ie+`
uniform sampler2D uSource;
uniform float uAmount;
void main(){outColor=vec4(texture(uSource,here()).x*uAmount,0.,0.,1.);}`,To=Ie+`
uniform sampler2D uPressure, uDivergence;
void main(){
  vec2 uv=here();
  float l=texture(uPressure,uv-vec2(uTexel.x,0.)).x,r=texture(uPressure,uv+vec2(uTexel.x,0.)).x;
  float t=texture(uPressure,uv+vec2(0.,uTexel.y)).x,b=texture(uPressure,uv-vec2(0.,uTexel.y)).x;
  outColor=vec4((l+r+b+t-texture(uDivergence,uv).x)*.25,0.,0.,1.);
}`,Mo=Ie+`
uniform sampler2D uPressure, uVelocity;
void main(){
  vec2 uv=here();
  float l=texture(uPressure,uv-vec2(uTexel.x,0.)).x,r=texture(uPressure,uv+vec2(uTexel.x,0.)).x;
  float t=texture(uPressure,uv+vec2(0.,uTexel.y)).x,b=texture(uPressure,uv-vec2(0.,uTexel.y)).x;
  outColor=vec4(texture(uVelocity,uv).xy-vec2(r-l,t-b),0.,1.);
}`,Ro=Ie+`
uniform sampler2D uVelocity;
uniform float uDt, uDissipation;
void main(){
  vec2 uv=here();
  vec2 back=uv-uDt*texture(uVelocity,uv).xy*uTexel;
  outColor=vec4(texture(uVelocity,back).xy/(1.+uDissipation*uDt),0.,1.);
}`,tt=new Set,Ue=0,kn=0,br=null,ze=[],Ze=0,mt=1,Ft=0,Er=0;function ko(e){Ue=0,Ze&&e>Ze&&(ze.push(e-Ze),ze.length>120&&ze.shift()),Ze=e,kn=performance.now();const n=[...tt];tt.clear();for(const l of n)try{l(e)}catch(i){typeof console<"u"&&console.error(i)}he==null||he.endFrame(e);const r=performance.now()-kn;if(Ft=.9*Ft+.1*r,e-Er>500){Er=e;const l=Xr(),i=!!(he!=null&&he.overloaded);(Ft>.5*l||i)&&mt<3?mt++:Ft<.2*l&&!i&&mt>1&&mt--}tt.size||(Ze=0)}var Gr=()=>mt;function An(e){tt.add(e),Ue&&br!==requestAnimationFrame&&(Ue=0),Ue||(br=requestAnimationFrame,Ue=requestAnimationFrame(ko))}function Sn(e){tt.delete(e),!tt.size&&Ue&&(cancelAnimationFrame(Ue),Ue=0,Ze=0)}function Xr(){if(ze.length<20)return 1e3/60;const e=[...ze].sort((n,r)=>n-r);return Math.min(1e3/60,Math.max(1e3/360,e[Math.floor(.2*e.length)]))}var pt=new Set,Pn=new Set,Cn=new Set,Un=new Set,In=new Set,vn=!1,Tr=null,Pe=null,Ce=null,Re=null,Mr=e=>{let n=null;for(const r of pt){if(r.target===window){(!r.near||r.near()||r.inside)&&r.move(e);continue}n??(n=e.composedPath()),n.includes(r.target)?(r.inside=!0,r.move(e)):r.inside&&(r.inside=!1,r.leave())}},Ao=()=>{for(const e of pt)e.inside=!1,e.leave()},So=()=>{for(const e of[...Pn])e()},Fo=()=>{for(const e of[...Cn])e()},Rr=()=>{for(const e of[...Un])e()},Lo=e=>{for(const n of[...In])n(e)};function kr(){var r,l,i,a,m,v,f,p;const e=pt.size+Pn.size+Cn.size+Un.size+In.size>0;if(typeof document>"u"||(Tr!==document&&(vn=!1,Tr=document,Pe=Ce=null,Re=null),e===vn))return;vn=e;const n=e?"addEventListener":"removeEventListener";if(typeof window<"u"&&((r=window[n])==null||r.call(window,"pointermove",Mr,{capture:!0,passive:!0}),(l=window[n])==null||l.call(window,"pointerdown",Mr,{capture:!0,passive:!0})),(i=document[n])==null||i.call(document,"pointerleave",Ao),(a=document[n])==null||a.call(document,"visibilitychange",So),(v=(m=typeof document>"u"?void 0:document.fonts)==null?void 0:m[n])==null||v.call(m,"loadingdone",Lo),Pe??(Pe=typeof matchMedia=="function"?matchMedia("(prefers-reduced-motion: reduce)"):null),Ce??(Ce=typeof matchMedia=="function"?matchMedia("(prefers-color-scheme: dark)"):null),(f=Pe==null?void 0:Pe[n])==null||f.call(Pe,"change",Fo),(p=Ce==null?void 0:Ce[n])==null||p.call(Ce,"change",Rr),e){Re??(Re=typeof MutationObserver=="function"?new MutationObserver(Rr):null);for(const h of[document.documentElement,document.body])h&&(Re==null||Re.observe(h,{attributes:!0,attributeFilter:["class","data-theme","data-mode","style"]}))}else Re==null||Re.disconnect()}var ot=(e,n)=>(e.add(n),kr(),()=>{e.delete(n)&&kr()});function Or(e){var m;const n=e.target;if(typeof window>"u"||n===window||typeof document>"u")return ot(pt,e);const r=n,l=(m=r.getRootNode)==null?void 0:m.call(r);if(r.ownerDocument===document&&(!l||l===document))return ot(pt,e);const i=e.move,a=()=>e.leave();return r.addEventListener("pointermove",i,{passive:!0}),r.addEventListener("pointerdown",i,{passive:!0}),r.addEventListener("pointerleave",a),()=>{r.removeEventListener("pointermove",i),r.removeEventListener("pointerdown",i),r.removeEventListener("pointerleave",a)}}var Hr=e=>ot(Pn,e),zr=e=>ot(Cn,e),_o=e=>ot(Un,e),dt=new Map,xe=null,xn=new Map,Nr=e=>n=>{for(const r of n)for(const l of[...e.get(r.target)??[]])l(r)},Vr=(e,n,r)=>{const l=e.get(n);return l?(l.add(r),!1):(e.set(n,new Set([r])),!0)},jr=(e,n,r)=>{const l=e.get(n);return!(!(l!=null&&l.delete(r))||l.size)&&(e.delete(n),!0)};function yn(e,n){return typeof ResizeObserver!="function"?()=>{}:(xe??(xe=new ResizeObserver(Nr(dt))),Vr(dt,e,n)?xe.observe(e):queueMicrotask(()=>{var r;(r=dt.get(e))!=null&&r.has(n)&&n({target:e})}),()=>{jr(dt,e,n)&&(xe==null||xe.unobserve(e)),dt.size||(xe==null||xe.disconnect(),xe=null)})}var Do=268435456,Lt=[1,.85,.7,.58,.5],Po=class{constructor(){this.lost=!1,this.level=0,this.programs=new Map,this.atlases=new Map,this.vertex=null,this.vao=null,this.clients=new Set,this.parallel=null,this.timer=null,this.queries=[],this.open=null,this.drew=!1,this.gpuMs=[],this.cpuMs=[],this.changedAt=0,this.lostTimer=0,this.renderer="",this.overloaded=!1,this.onLost=e=>{e.preventDefault(),this.lost=!0,this.reset();for(const n of[...this.clients])n.lost();clearTimeout(this.lostTimer),this.lostTimer=setTimeout(()=>{var n,r,l,i;if(this.lost){(r=(n=this.canvas).removeEventListener)==null||r.call(n,"webglcontextlost",this.onLost),(i=(l=this.canvas).removeEventListener)==null||i.call(l,"webglcontextrestored",this.onRestored);try{this.create()}catch{return}this.restored()}},1e3)},this.onRestored=()=>{clearTimeout(this.lostTimer),this.lost=!1,this.restored()},this.idleTimer=0,this.create()}create(){var a,m;let e=null,n=null;const r={alpha:!0,antialias:!1,depth:!1,stencil:!1,premultipliedAlpha:!0,preserveDrawingBuffer:!0,powerPreference:"high-performance"};if(typeof document<"u"&&(e=document.createElement("canvas"),n=e.getContext("webgl2",r)),!n&&typeof OffscreenCanvas=="function")try{e=new OffscreenCanvas(1,1),n=e.getContext("webgl2",r)}catch{n=null}if(!n||!e)throw new Error("WebGL2 is not available for the glyph field.");this.canvas=e,this.gl=n,this.lost=!1,e.width=e.height=1,this.parallel=n.getExtension("KHR_parallel_shader_compile"),this.timer=n.getExtension("EXT_disjoint_timer_query_webgl2");const l=n.getExtension("WEBGL_debug_renderer_info");this.renderer=String(n.getParameter(l?l.UNMASKED_RENDERER_WEBGL:n.RENDERER)??""),(a=e.addEventListener)==null||a.call(e,"webglcontextlost",this.onLost),(m=e.addEventListener)==null||m.call(e,"webglcontextrestored",this.onRestored);const i=!n.getExtension("EXT_color_buffer_float")&&!n.getExtension("EXT_color_buffer_half_float");this.program(Br(i)),this.program(qr)}restored(){for(const e of[...this.clients])e.restored()}reset(){this.programs.clear(),this.atlases.clear(),this.vertex=null,this.vao=null,this.queries=[],this.open=null,this.gpuMs=[],this.cpuMs=[]}join(e){this.clients.add(e),clearTimeout(this.idleTimer)}leave(e){this.clients.delete(e),this.clients.size?this.fit():(this.lost||(this.canvas.width=this.canvas.height=1,this.level=0),clearTimeout(this.idleTimer),this.idleTimer=setTimeout(()=>{this.clients.size||this.dispose()},3e3))}dispose(){var e,n,r,l,i;if(clearTimeout(this.lostTimer),(n=(e=this.canvas).removeEventListener)==null||n.call(e,"webglcontextlost",this.onLost),(l=(r=this.canvas).removeEventListener)==null||l.call(r,"webglcontextrestored",this.onRestored),!this.lost)try{(i=this.gl.getExtension("WEBGL_lose_context"))==null||i.loseContext()}catch{}this.reset(),he===this&&(he=null)}program(e){const n=this.programs.get(e);if(n)return n;const r=this.gl,l=(v,f)=>{const p=r.createShader(v);return r.shaderSource(p,f),r.compileShader(p),p};this.vertex??(this.vertex=l(r.VERTEX_SHADER,`#version 300 es
void main(){vec2 p=vec2((gl_VertexID<<1)&2,gl_VertexID&2);gl_Position=vec4(p*2.-1.,0.,1.);}`));const i=r.createProgram(),a=l(r.FRAGMENT_SHADER,e);r.attachShader(i,this.vertex),r.attachShader(i,a),r.linkProgram(i);const m={program:i,locations:null,fragment:a};return this.programs.set(e,m),m}linked(e,n=!1){if(e.locations)return!0;const r=this.gl;if(!n&&this.parallel&&!r.getProgramParameter(e.program,this.parallel.COMPLETION_STATUS_KHR))return!1;if(!r.getProgramParameter(e.program,r.LINK_STATUS)){if(r.isContextLost())return!1;throw new Error(e.fragment&&r.getShaderInfoLog(e.fragment)||r.getProgramInfoLog(e.program)||"Glyph field program failed to link.")}const l={};for(let i=r.getProgramParameter(e.program,r.ACTIVE_UNIFORMS)-1;i>=0;i--){const a=r.getActiveUniform(e.program,i).name.replace(/\[0\]$/,"");l[a]=r.getUniformLocation(e.program,a)}return e.locations=l,!0}get canCompileInParallel(){return!!this.parallel}vertexArray(){return this.vao??(this.vao=this.gl.createVertexArray())}atlas(e,n,r){let l=this.atlases.get(e);if(l)l.refs||(this.atlases.delete(e),this.atlases.set(e,l));else{const i=this.gl,a=i.createTexture();i.bindTexture(i.TEXTURE_2D,a);for(const[m,v]of[[i.TEXTURE_MIN_FILTER,i.NEAREST],[i.TEXTURE_MAG_FILTER,i.NEAREST],[i.TEXTURE_WRAP_S,i.CLAMP_TO_EDGE],[i.TEXTURE_WRAP_T,i.CLAMP_TO_EDGE]])i.texParameteri(i.TEXTURE_2D,m,v);l={key:e,texture:a,refs:0,final:!1,users:new Set,bytes:0},this.atlases.set(e,l),this.paintAtlas(l,n)}return l.refs++,l.users.add(r),l}hasAtlas(e){return this.atlases.has(e)}paintAtlas(e,n){const r=this.gl,{image:l,final:i}=n();r.bindTexture(r.TEXTURE_2D,e.texture),r.pixelStorei(r.UNPACK_PREMULTIPLY_ALPHA_WEBGL,!1),r.pixelStorei(r.UNPACK_FLIP_Y_WEBGL,!1),r.texImage2D(r.TEXTURE_2D,0,r.RGBA8,r.RGBA,r.UNSIGNED_BYTE,l),e.bytes=l.width*l.height*4,e.final=i;for(const a of[...e.users])a()}releaseAtlas(e,n){e&&e.users.delete(n)&&(--e.refs>0||this.evictAtlases())}evictAtlases(){const e=[...this.atlases.values()].filter(r=>!r.refs);let n=e.length-32;for(const r of e){if(n<=0&&!this.overBudget)break;n--,this.atlases.delete(r.key),this.lost||this.gl.deleteTexture(r.texture)}}fit(){if(this.lost)return;let e=1,n=1;for(const i of this.clients){const[a,m]=i.size();e=Math.max(e,a),n=Math.max(n,m)}const r=this.canvas,l=i=>256*Math.ceil(i/256);r.width*r.height>l(e)*l(n)*2&&(r.width=l(e),r.height=l(n))}get bytes(){let e=this.canvas.width*this.canvas.height*4*2;for(const n of this.clients)e+=n.bytes();for(const n of this.atlases.values())e+=n.bytes;return e}get overBudget(){return this.bytes>Do}get roomy(){return this.bytes<2281701376e-1}begin(e,n){const r=this.canvas;if(r.width<e||r.height<n){const l=i=>256*Math.ceil(i/256);r.width=Math.max(r.width,l(e)),r.height=Math.max(r.height,l(n))}if(!this.drew&&(this.drew=!0,this.timer&&!this.open&&this.queries.length<4)){const l=this.gl.createQuery();l&&(this.gl.beginQuery(this.timer.TIME_ELAPSED_EXT,l),this.open=l)}}present(e,n,r){e.globalCompositeOperation="copy",e.imageSmoothingEnabled=!1,e.drawImage(this.canvas,0,this.canvas.height-r,n,r,0,0,n,r)}endFrame(e){if(this.lost)return void(this.drew=!1);const n=this.gl,r=this.timer;if(this.drew&&(this.cpuMs.push(performance.now()-kn),this.cpuMs.length>60&&this.cpuMs.shift(),r&&this.open&&(n.endQuery(r.TIME_ELAPSED_EXT),this.queries.push({query:this.open,frame:e}),this.open=null)),this.drew=!1,r){const l=n.getParameter(r.GPU_DISJOINT_EXT);for(;this.queries.length&&n.getQueryParameter(this.queries[0].query,n.QUERY_RESULT_AVAILABLE);){const{query:i}=this.queries.shift();l||(this.gpuMs.push(n.getQueryParameter(i,n.QUERY_RESULT)/1e6),this.gpuMs.length>60&&this.gpuMs.shift()),n.deleteQuery(i)}}this.adapt(e)}adapt(e){const n=Xr(),r=this.gpuMs.length>=5&&Math.min(...this.gpuMs.slice(-5))>1.5*n,l=this.gpuMs.length>=10||r?this.gpuMs:null;let i;if(l)i=[...l].sort((m,v)=>m-v)[Math.floor(.75*l.length)]/n;else{if(ze.length<30||this.cpuMs.length<10)return;const m=ze.slice(-30).sort((f,p)=>f-p),v=[...this.cpuMs].sort((f,p)=>f-p)[this.cpuMs.length>>1];i=m[Math.floor(.75*m.length)]/n>1.6&&v<.5*n?1:.5}if(this.overloaded=i>.8&&this.atMinimum,e-this.changedAt<(i>.8?800:4e3))return;let a=this.level;if(i>2?a=Lt.length-1:i>.8&&a<Lt.length-1?a++:i<.3&&a>0&&a--,a!==this.level){this.level=a,this.changedAt=e,this.gpuMs=[],this.cpuMs=[];for(const m of[...this.clients])m.rescale()}}get scale(){return Lt[this.level]}get atMinimum(){return this.level===Lt.length-1}},he=null,Ar=!1;function Bn(){if(he||Ar)return he;try{he=new Po}catch{Ar=!0}return he}var $r=()=>he,Wr=new WeakSet,Yr=e=>Wr.has(e),Co=class{constructor(e,n,r){this.previous=null,this.peak=0,this.width=e,this.height=n,this.span=r??Math.min(e,n);const l=e*n;this.pixels=new Uint8Array(4*l),this.signal=new Float32Array(l),this.offsetX=new Float32Array(l),this.offsetY=new Float32Array(l),this.driftX=new Float32Array(l),this.driftY=new Float32Array(l),this.u=new Float32Array(l),this.v=new Float32Array(l),this.ink=new Float32Array(l),this.a=new Float32Array(l),this.b=new Float32Array(l),this.c=new Float32Array(l),this.pressure=new Float32Array(l),this.pressureNext=new Float32Array(l),this.divergence=new Float32Array(l),this.curl=new Float32Array(l)}get active(){return this.peak>2e-4}clear(){for(const e of[this.u,this.v,this.ink,this.a,this.b,this.c,this.pressure,this.pressureNext,this.divergence,this.curl,this.signal,this.pixels,this.offsetX,this.offsetY,this.driftX,this.driftY])e.fill(0);this.previous=null,this.peak=0}leave(){this.previous=null}move(e,n,r,l=performance.now()){if(!Number.isFinite(e+n+r+l))return;e=Math.max(0,Math.min(1,e)),n=Math.max(0,Math.min(1,n)),r=Math.max(.1,Math.min(1,r));const i=this.previous;if(this.previous={x:e,y:n,time:l},!i)return;const a=(e-i.x)*(this.width-1),m=(n-i.y)*(this.height-1),v=Math.hypot(a,m);if(v<.001)return;const f=this.span,p=l>i.time?Math.max(1/240,(l-i.time)/1e3):1/60,h=Math.min(12,v/f/p),b=1-Math.exp(-h/2.2),t=f*(.022+.075*r)*(.65+.95*b),R=Math.min(128,Math.max(1,Math.ceil(v/(.4*t)))),V=v/t*(.32+.22*b)/R,B=a/v,d=m/v,A=-d,X=B,_=v*(2+18*b)/R;for(let D=0;D<R;D++){const $=(D+.5)/R,re=(i.x+(e-i.x)*$)*(this.width-1),ae=(i.y+(n-i.y)*$)*(this.height-1),O=Math.max(1,Math.floor(re-3*t)),W=Math.min(this.width-2,Math.ceil(re+3*t)),we=Math.max(1,Math.floor(ae-3*t)),pe=Math.min(this.height-2,Math.ceil(ae+3*t));for(let fe=we;fe<=pe;fe++)for(let g=O;g<=W;g++){const w=((g-re)*B+(fe-ae)*d)/t,C=((g-re)*A+(fe-ae)*X)/t,ne=Math.exp(-w*w*.65-C*C*1.7),q=fe*this.width+g;this.ink[q]=1-(1-this.ink[q])*Math.exp(-ne*V);const ee=1-C*C*.55,j=w*C*.7;this.u[q]=Math.max(-80,Math.min(80,this.u[q]+(B*ee+A*j)*ne*_)),this.v[q]=Math.max(-80,Math.min(80,this.v[q]+(d*ee+X*j)*ne*_))}}this.peak=1}read(e,n,r){const l=0|(n=Math.max(.5,Math.min(this.width-1.5,n))),i=0|(r=Math.max(.5,Math.min(this.height-1.5,r))),a=n-l,m=r-i,v=i*this.width+l;return(e[v]*(1-a)+e[v+1]*a)*(1-m)+(e[v+this.width]*(1-a)+e[v+this.width+1]*a)*m}sample(e,n){return this.read(this.signal,e*(this.width-1),n*(this.height-1))}displacement(e,n,r){r[0]=this.read(this.offsetX,e*(this.width-1),n*(this.height-1)),r[1]=this.read(this.offsetY,e*(this.width-1),n*(this.height-1))}step(e){if(!this.active||!Number.isFinite(e))return;let n=Math.max(0,Math.min(.05,e));const r=this.width,l=this.height;for(;n>1e-7;){const i=Math.min(.016666666666666666,n);n-=i;const a=Math.exp(-1.9*i),m=Math.exp(-3.2*i);for(let f=1;f<l-1;f++)for(let p=1;p<r-1;p++){const h=f*r+p,b=Math.max(.5,Math.min(r-1.5,p-this.u[h]*i)),t=Math.max(.5,Math.min(l-1.5,f-this.v[h]*i)),R=0|b,V=0|t,B=b-R,d=t-V,A=V*r+R,X=(1-B)*(1-d),_=B*(1-d),D=(1-B)*d,$=B*d;this.a[h]=(this.u[A]*X+this.u[A+1]*_+this.u[A+r]*D+this.u[A+r+1]*$)*a,this.b[h]=(this.v[A]*X+this.v[A+1]*_+this.v[A+r]*D+this.v[A+r+1]*$)*a,this.c[h]=(this.ink[A]*X+this.ink[A+1]*_+this.ink[A+r]*D+this.ink[A+r+1]*$)*m}[this.u,this.a]=[this.a,this.u],[this.v,this.b]=[this.b,this.v],[this.ink,this.c]=[this.c,this.ink];for(let f=1;f<l-1;f++)for(let p=1;p<r-1;p++){const h=f*r+p;this.curl[h]=.5*(this.v[h+1]-this.v[h-1]-this.u[h+r]+this.u[h-r])}for(let f=1;f<l-1;f++)for(let p=1;p<r-1;p++){const h=f*r+p,b=Math.abs(this.curl[h+1])-Math.abs(this.curl[h-1]),t=Math.abs(this.curl[h+r])-Math.abs(this.curl[h-r]),R=Math.hypot(b,t)+1e-4,V=this.curl[h]*i*10,B=Math.hypot(this.u[h],this.v[h]);this.u[h]+=t/R*V,this.v[h]-=b/R*V;const d=Math.min(1,B/(Math.hypot(this.u[h],this.v[h])+1e-6));this.u[h]*=d,this.v[h]*=d,this.divergence[h]=-.5*(this.u[h+1]-this.u[h-1]+this.v[h+r]-this.v[h-r])}this.pressure.fill(0);for(let f=0;f<4;f++){for(let p=1;p<l-1;p++)for(let h=1;h<r-1;h++){const b=p*r+h;this.pressureNext[b]=.25*(this.divergence[b]+this.pressure[b-1]+this.pressure[b+1]+this.pressure[b-r]+this.pressure[b+r])}[this.pressure,this.pressureNext]=[this.pressureNext,this.pressure]}let v=0;for(let f=1;f<l-1;f++)for(let p=1;p<r-1;p++){const h=f*r+p;this.u[h]=Math.max(-80,Math.min(80,this.u[h]-.5*(this.pressure[h+1]-this.pressure[h-1]))),this.v[h]=Math.max(-80,Math.min(80,this.v[h]-.5*(this.pressure[h+r]-this.pressure[h-r]))),this.driftX[h]+=(3.6*this.u[h]-58*this.offsetX[h]-12*this.driftX[h])*i,this.driftY[h]+=(3.6*this.v[h]-58*this.offsetY[h]-12*this.driftY[h])*i,this.offsetX[h]=Math.max(-2,Math.min(2,this.offsetX[h]+this.driftX[h]*i)),this.offsetY[h]=Math.max(-2,Math.min(2,this.offsetY[h]+this.driftY[h]*i)),this.signal[h]=1-Math.exp(.06*-Math.hypot(this.u[h],this.v[h])-.16*this.ink[h]),v=Math.max(v,this.signal[h])}this.peak=v}if(this.active)for(let i=0;i<this.ink.length;i++){const a=Math.round(65535*Math.min(1,this.signal[i]));this.pixels[4*i]=a>>>8,this.pixels[4*i+1]=255&a,this.pixels[4*i+2]=Math.round(128+63.5*this.offsetX[i]),this.pixels[4*i+3]=Math.round(128+63.5*this.offsetY[i])}else this.clear()}};function Sr(e,n,r,l,i){if(!(i<1))for(let a=0;a<2;a++){for(let m=0;m<l;m++){let v=0;const f=m*r;for(let p=-i;p<=i;p++)v+=e[f+Math.max(0,Math.min(r-1,p))];for(let p=0;p<r;p++)n[f+p]=v/(2*i+1),v+=e[f+Math.min(r-1,p+i+1)]-e[f+Math.max(0,p-i)]}for(let m=0;m<r;m++){let v=0;for(let f=-i;f<=i;f++)v+=n[Math.max(0,Math.min(l-1,f))*r+m];for(let f=0;f<l;f++)e[f*r+m]=v/(2*i+1),v+=n[Math.min(l-1,f+i+1)*r+m]-n[Math.max(0,f-i)*r+m]}}}var wn=new Float32Array(0),_t,Kr=["none","flow","orb","drift","swell","caustics","contour","aurora","ripple","halo","interference","mesh","cascade","constellation"],It=["glyph","dot","pixel","mosaic","lego","voxel","disco","dither","hex","led","cmyk","stipple","engraving","crosshatch","woodblock","risograph","pointillism"],Qr=[{value:"fluid",label:"Fluid",description:"An ink-in-water wake that bends the image and swirls the characters."},{value:"lantern",label:"Lantern",description:"A soft light follows the pointer and lifts the characters it passes, without distortion."},{value:"trail",label:"Trail",description:"The Asciify Trail: in the wake, sparse characters fill in and dense ones open up, then settle. No distortion."},{value:"loupe",label:"Loupe",description:"A glass loupe glides after the pointer and magnifies the characters it covers."},{value:"reveal",label:"Reveal",description:"Under a soft spotlight the characters take the source image's own colours. Best over a photo or video in one ink."},{value:"none",label:"Off",description:"The field stays still under the pointer."}],Jr={radius:.13,strength:.5,trail:.45,follow:12},Zr={strength:.55,radius:.45},ye={radius:.075,force:1,dissipation:1.6,refraction:.03,glow:.55,resolution:128,curl:24},eo={charset:" .,:-=+*#%@",cellSize:12,cellAspect:1,cellFit:"pixel",font:'ui-monospace, "SF Mono", Menlo, Consolas, monospace',fontWeight:500,glyphScale:.8,mark:"glyph",dotScale:.9,markAngle:-35,glyphOrder:"tone",color:"ink",ink:"#f3ba32",highlight:"auto",background:"transparent",brightness:0,contrast:1,gamma:1,invert:!1,levels:"auto",blend:"auto",saturation:1.1,shade:.5,backdrop:0,fit:"cover",focus:{x:.5,y:.5},vignette:0,drift:0,scene:"flow",scale:1,clear:!1,speed:1,hover:"fluid",fluid:ye,lantern:Jr,trail:Zr,scope:"all",maxPixelRatio:2,respectReducedMotion:!0},I=(e,n,r,l)=>typeof e=="number"&&Number.isFinite(e)?Math.min(r,Math.max(n,e)):l,bn=(e,n)=>typeof e=="string"&&/^#([\da-f]{3}|[\da-f]{6})$/i.test(e.trim())?e.trim().toLowerCase():n,Fr=/^[\u0020-\u007e\u00a0-\u02ff\u0370-\u0482\u048a-\u052f\u2010-\u20cf\u2100-\u2bff]*$/;function Ne(e){if(!Fr.test(e)&&_t===void 0){const n=Intl.Segmenter;_t=n?new n(void 0,{granularity:"grapheme"}):null}return(_t&&!Fr.test(e)?Array.from(_t.segment(e),n=>n.segment):Array.from(e)).filter(n=>n!==`
`&&n!=="\r"&&n!=="	").slice(0,256)}function Bt(e={},n=eo){var b,t,R,V,B,d,A,X;const r=typeof e.charset=="string"&&Ne(e.charset).length>=2?e.charset:n.charset,l=n.fluid||ye,i=e.fluid===void 0?n.fluid:e.fluid,a=i!==!1&&(()=>{const _=i!==!0&&i?{...l,...i}:l;return{radius:I(_.radius,.005,.5,ye.radius),force:I(_.force,0,8,ye.force),dissipation:I(_.dissipation,.05,20,ye.dissipation),refraction:I(_.refraction,0,.5,ye.refraction),glow:I(_.glow,0,2,ye.glow),resolution:Math.round(I(_.resolution,16,512,ye.resolution)),curl:I(_.curl,0,80,ye.curl)}})(),m=Kr.includes(e.scene)?e.scene:n.scene,v=!!e.clear&&typeof e.clear=="object"&&("element"in e.clear||"nodeType"in e.clear),f=e.clear===void 0||v?n.clear:e.clear,p=!!f&&{x:I(f.x,-.5,1.5,.25),y:I(f.y,-.5,1.5,.3),width:I(f.width,0,2,.5),height:I(f.height,0,2,.4),feather:I(f.feather,.005,1,.12),shape:f.shape==="rect"?"rect":"ellipse"};return{charset:r,cellSize:I(e.cellSize,3,96,n.cellSize),cellAspect:I(e.cellAspect,.3,2,n.cellAspect),cellFit:e.cellFit==="fill"||e.cellFit==="pixel"?e.cellFit:n.cellFit,font:typeof e.font=="string"&&e.font.trim()?e.font:n.font,fontWeight:Math.round(I(e.fontWeight,100,900,n.fontWeight)),glyphScale:I(e.glyphScale,.3,1.6,n.glyphScale),mark:It.includes(e.mark)?e.mark:n.mark,dotScale:I(e.dotScale,.1,1.4,n.dotScale),markAngle:I(e.markAngle,-360,360,n.markAngle),glyphOrder:e.glyphOrder==="text"||e.glyphOrder==="tone"?e.glyphOrder:n.glyphOrder,color:e.color==="source"||e.color==="ink"?e.color:n.color,ink:bn(e.ink,n.ink),highlight:e.highlight==="auto"?"auto":bn(e.highlight,n.highlight),background:e.background==="transparent"?"transparent":bn(e.background,n.background),brightness:I(e.brightness,-1,1,n.brightness),contrast:I(e.contrast,0,4,n.contrast),gamma:I(e.gamma,.2,5,n.gamma),invert:typeof e.invert=="boolean"?e.invert:n.invert,levels:e.levels==="none"||e.levels==="auto"?e.levels:n.levels,blend:e.blend==="auto"||e.blend==="coarse"||e.blend==="on"||e.blend==="off"?e.blend:n.blend,saturation:I(e.saturation,0,3,n.saturation),shade:I(e.shade,0,1,n.shade),backdrop:I(e.backdrop,0,1,n.backdrop),fit:e.fit==="contain"||e.fit==="cover"?e.fit:n.fit,focus:{x:I((b=e.focus)==null?void 0:b.x,0,1,n.focus.x),y:I((t=e.focus)==null?void 0:t.y,0,1,n.focus.y)},vignette:I(e.vignette,0,1,n.vignette),drift:I(e.drift,0,2,n.drift),scene:m,scale:I(e.scale,.2,6,n.scale),clear:p,speed:I(e.speed,0,5,n.speed),hover:Qr.some(_=>_.value===e.hover)?e.hover:n.hover,fluid:a,lantern:{radius:I((R=e.lantern)==null?void 0:R.radius,.02,.6,n.lantern.radius),strength:I((V=e.lantern)==null?void 0:V.strength,0,1.5,n.lantern.strength),trail:I((B=e.lantern)==null?void 0:B.trail,0,3,n.lantern.trail),follow:I((d=e.lantern)==null?void 0:d.follow,1,60,n.lantern.follow)},trail:{strength:I((A=e.trail)==null?void 0:A.strength,0,1,n.trail.strength),radius:I((X=e.trail)==null?void 0:X.radius,.1,1,n.trail.radius)},scope:e.scope===void 0?n.scope:(h=e.scope,h==="subject"||h==="outside"?h:"all"),maxPixelRatio:I(e.maxPixelRatio,.5,4,n.maxPixelRatio),respectReducedMotion:typeof e.respectReducedMotion=="boolean"?e.respectReducedMotion:n.respectReducedMotion};var h}function Fn(e,n,r,l,i,a={x:.5,y:.5}){if(!(e>0&&n>0&&r>0&&l>0))return{scale:[1,1],offset:[0,0]};const m=e/n,v=r/l,f=v>m,p=i==="cover"?f?[m/v,1]:[1,v/m]:f?[1,v/m]:[m/v,1];return{scale:p,offset:i==="cover"?[(1-p[0])*a.x,(1-p[1])*a.y]:[.5*(1-p[0]),.5*(1-p[1])]}}function qt(e,n,r,l){const i=Math.max(2,Math.round(e*r)),a=Math.max(2,Math.round(i*n)),m=Math.min(16,Math.max(1,l)),v=Math.ceil(l/m);return{width:a,height:i,columns:m,rows:v,atlasWidth:m*a,atlasHeight:v*i}}function En(e){const n=e.replace("#",""),r=n.length===3?n.split("").map(i=>i+i).join(""):n,l=Number.parseInt(r,16);return Number.isFinite(l)&&r.length===6?[(l>>16&255)/255,(l>>8&255)/255,(255&l)/255]:[0,0,0]}function to(e){const n=new Uint32Array(256);let r=0;for(let m=0;m+3<e.length;m+=4)e[m+3]<8||(n[Math.round(.2126*e[m]+.7152*e[m+1]+.0722*e[m+2])]++,r++);if(!r)return[0,1];const l=m=>{let v=0;for(let f=0;f<256;f++)if(v+=n[f],v>=r*m)return f/255;return 1},i=l(.02),a=l(.98);return a-i<.08?[0,1]:[i,a]}var et=e=>e.hover!=="fluid"||e.fluid?e.hover:"none";function no(e,n,r=24,l=.12,i="ellipse"){return e.width>0&&e.height>0&&{x:(n.left-r-e.left)/e.width,y:(n.top-r-e.top)/e.height,width:(n.width+2*r)/e.width,height:(n.height+2*r)/e.height,feather:l,shape:i}}var Oe,Uo=/\.(mp4|webm|mov|m4v|ogv)(\?|#|$)/i,Lr=/swiftshader|llvmpipe|softpipe|software|basic render/i;function ro(){var e;if(Oe)return Oe;if(typeof document>"u"||globalThis.WebGL2RenderingContext===void 0)return Oe="none";try{const n=Bn();if(n)return Oe=Lr.test(n.renderer)?"software":"gpu";const r=document.createElement("canvas").getContext("webgl2");if(!r)return Oe="none";const l=r.getExtension("WEBGL_debug_renderer_info"),i=String(r.getParameter(l?l.UNMASKED_RENDERER_WEBGL:r.RENDERER)??"");return(e=r.getExtension("WEBGL_lose_context"))==null||e.loseContext(),Oe=Lr.test(i)?"software":"gpu"}catch{return Oe="none"}}var qn=(e,n)=>`${e.fontWeight} ${Math.max(1,n.height*e.glyphScale)}px ${e.font}`,oo=(e,n)=>`${Ne(e.charset).join("")}\0${qn(e,n)}\0${n.width}x${n.height}x${n.columns}`;function ao(e,n,r,l){const i=Math.min(globalThis.devicePixelRatio||1,e.maxPixelRatio,Math.sqrt(8294400/Math.max(1,n*r)));return l<1&&i>1?Math.max(1,i*l):i}function io(e,n,r){const l=Ne(e.charset),i=document.createElement("canvas");i.width=n.atlasWidth,i.height=n.atlasHeight;const a=i.getContext("2d");a.fillStyle="#fff",a.textAlign="center",a.textBaseline="alphabetic",a.font=qn(e,n);const m=a.measureText("M"),v=m.actualBoundingBoxAscent||n.height*e.glyphScale*.7,f=m.actualBoundingBoxDescent||0,p=(n.height+v-f)/2;l.forEach((t,R)=>{const V=R%n.columns*n.width+n.width/2,B=Math.floor(R/n.columns)*n.height+p,d=a.measureText(t).width;a.save(),a.beginPath(),a.rect(V-n.width/2,B-p,n.width,n.height),a.clip(),a.translate(V,B),d>.96*n.width&&a.scale(.96*n.width/d,1),a.fillText(t,0,0),a.restore()});let h=!0;const b=document.fonts;try{h=!(b!=null&&b.check)||b.check(a.font,l.join(""))}catch{}return r&&queueMicrotask(()=>{i.width=i.height=1}),{image:i,final:h}}var _r=()=>{};function lo(e,n={}){if(typeof document>"u"||e.ownerDocument!==document)return!1;const r=e.clientWidth,l=e.clientHeight;if(!r||!l)return!1;const i=Bn();if(!i||i.lost)return!1;const{clear:a,source:m,interactionTarget:v,onError:f,label:p,...h}=n,b=Bt(h),t=qt(b.cellSize,b.cellAspect,ao(b,r,l,i.scale),Ne(b.charset).length),R=oo(b,t);if(i.hasAtlas(R))return!1;try{i.releaseAtlas(i.atlas(R,()=>io(b,t,!0),_r),_r)}catch{return!1}return!0}function Io(e={}){const n=ro();return n==="gpu"||n==="software"&&!!e.allowSoftware}function so(e,n={}){const r=o=>typeof HTMLElement<"u"&&o instanceof HTMLElement,l=o=>{const c=o.clear;return r(c)?{element:{element:c},rest:{...o,clear:void 0}}:c&&typeof c=="object"&&r(c.element)?{element:c,rest:{...o,clear:void 0}}:{element:null,rest:o}};let i=l(n).element,a=Bt(l(n).rest);const m=n.onError,v=e.ownerDocument,f=v&&typeof document<"u"&&v!==document?null:Bn(),p=f?e.getContext("2d",{alpha:!0}):null,h=!(!f||!p),b=h?f.gl:e.getContext("webgl2",{alpha:!0,antialias:!1,depth:!1,stencil:!1,premultipliedAlpha:!0,powerPreference:"high-performance"});if(!b)throw new Error("WebGL2 is not available for the glyph field.");let t=b;h&&(o=>{Wr.add(o)})(e),e.setAttribute("role","img"),e.setAttribute("aria-label",n.label??"ASCII artwork"),e.dataset.renderer="glyph-field";let R=!1;const V=h?null:t.getExtension("KHR_parallel_shader_compile");let B,d,A=null,X=null,_=null,D=null,$=null,re=[],ae=[],O=null,W=!1,we=null,pe=null;const fe=[];let g=0,w=0,C=1,ne=0,q=qt(a.cellSize,a.cellAspect,1,2),ee=2,j=0,ue=0,oe=0,S=null,me=!1,be=null,Ee=0,ke=0,at=0,je=null,it=0,Ht=0,vt=!0,lt=0,st=[0,1],On=-1/0,Hn=-1/0,zn=!1;const mo=(()=>{var c;const o=typeof navigator>"u"?void 0:navigator;return 1e3/(o&&(o.hardwareConcurrency>0&&o.hardwareConcurrency<=2||o.deviceMemory&&o.deviceMemory<=2||(c=o.connection)!=null&&c.saveData)?30:60)-2})();let Ae=!1,zt=0,Nn=0,Q=!0,Se=!1,Fe=!1,xt=!1,yt=!1,wt=!0,Le=null,_e=null,Nt=1/60,ct=0,Be=0,Te=[];const $e=new Float32Array(16),Vt=()=>{if(!i)return;const{element:o,padding:c=32,feather:u=.16,shape:s="rect"}=i,x=no(e.getBoundingClientRect(),o.getBoundingClientRect(),c,u,s),E=a.clear;JSON.stringify(E)!==JSON.stringify(x)&&(a={...a,clear:x},Q=!0)},jt=o=>o==="lantern"||o==="loupe"||o==="reveal",J={target:null,at:null,strength:0,trail:[]},ut=new Float32Array(36);let te=null,De=null,bt=!1;const Vn=()=>{if(!g||!w)return null;const o=g/w,c=Math.round(o>=1?128:128*o),u=Math.round(o>=1?128/o:128);return te&&te.width===c&&te.height===u||(te=new Co(c,u),De||(De=Ye(t.LINEAR)),t.bindTexture(t.TEXTURE_2D,De),t.texImage2D(t.TEXTURE_2D,0,t.RGBA8,c,u,0,t.RGBA,t.UNSIGNED_BYTE,te.pixels),bt=!0),te};let $t=0;const Wt=typeof matchMedia=="function"?matchMedia("(prefers-reduced-motion: reduce)"):null,We=()=>a.respectReducedMotion&&!!(Wt!=null&&Wt.matches);let Yt;const go=new Promise(o=>{Yt=o});let qe=!0;const ht=o=>m==null?void 0:m(o instanceof Error?o:new Error(String(o))),jn=(o,c)=>{const u=t.createShader(o);return fe.push(u),t.shaderSource(u,c),t.compileShader(u),u},Et=o=>{if(h)return f.program(o);pe??(pe=jn(t.VERTEX_SHADER,`#version 300 es
void main(){vec2 p=vec2((gl_VertexID<<1)&2,gl_VertexID&2);gl_Position=vec4(p*2.-1.,0.,1.);}`));const c=t.createProgram(),u=jn(t.FRAGMENT_SHADER,o);return t.attachShader(c,pe),t.attachShader(c,u),t.linkProgram(c),{program:c,locations:null,fragment:u}},Tt=(o,c=!1)=>{if(o.locations)return!0;if(h)return f.linked(o,c);const{program:u,fragment:s}=o;if(!c&&V&&!t.getProgramParameter(u,V.COMPLETION_STATUS_KHR))return!1;if(!t.getProgramParameter(u,t.LINK_STATUS)){if(t.isContextLost())return!1;throw new Error(t.getShaderInfoLog(s)||t.getProgramInfoLog(u)||"Glyph field program failed to link.")}const x={};for(let E=t.getProgramParameter(u,t.ACTIVE_UNIFORMS)-1;E>=0;E--){const M=t.getActiveUniform(u,E).name.replace(/\[0\]$/,"");x[M]=t.getUniformLocation(u,M)}return o.locations=x,!0},Kt=(o=!1)=>!!A&&!!X&&Tt(A,o)&&Tt(X,o),Ye=o=>{const c=t.createTexture();return t.bindTexture(t.TEXTURE_2D,c),t.texParameteri(t.TEXTURE_2D,t.TEXTURE_MIN_FILTER,o),t.texParameteri(t.TEXTURE_2D,t.TEXTURE_MAG_FILTER,o===t.LINEAR_MIPMAP_LINEAR?t.LINEAR:o),t.texParameteri(t.TEXTURE_2D,t.TEXTURE_WRAP_S,t.CLAMP_TO_EDGE),t.texParameteri(t.TEXTURE_2D,t.TEXTURE_WRAP_T,t.CLAMP_TO_EDGE),c},Qt=()=>{$&&(t.deleteTexture($.ink),t.deleteTexture($.under),t.deleteFramebuffer($.buffer),$=null)},$n=()=>(function(o,c,u,s,x,E,M){if(o!=="fill"||!(x>0&&E>0))return[M.width,M.height];const y=c*s,H=y*u;return[x/Math.max(1,Math.floor(x/H+1e-6)),E/Math.max(1,Math.floor(E/y+1e-6))]})(a.cellFit,a.cellSize,a.cellAspect,C,g,w,q),Jt=()=>!!a.fluid&&et(a)==="fluid",Zt=()=>{O==null||O.release(),W=!1,re.forEach(o=>t.deleteTexture(o)),ae.forEach(o=>t.deleteFramebuffer(o)),re=[],ae=[]};function Wn(){_&&!h&&t.deleteProgram(_.program),_=Et((o=>`#version 300 es
precision highp float;
precision highp sampler2D;
uniform sampler2D uPrevious;
uniform vec2 uSize;
uniform float uAspect, uDt, uDecay, uEnergyDecay, uRadius, uSplat;
uniform vec2 uFrom, uTo, uForce;
out vec4 outColor;
${Ir(o)}
float segment(vec2 p,vec2 a,vec2 b){vec2 pa=p-a,ba=b-a;float h=clamp(dot(pa,ba)/max(dot(ba,ba),1e-8),0.,1.);return length(pa-ba*h);}
void main(){
  vec2 uv=gl_FragCoord.xy/uSize;
  vec4 here=decodeFluid(texture(uPrevious,uv));
  vec4 state=decodeFluid(texture(uPrevious,uv-uDt*here.xy*vec2(1./uAspect,1.)));
  state.xy*=exp(-uDt*uDecay);
  state.z*=exp(-uDt*uEnergyDecay);
  if(uSplat>.5){
    vec2 scale=vec2(uAspect,1.);
    float d=segment(uv*scale,uFrom*scale,uTo*scale);
    float g=exp(-d*d/(uRadius*uRadius));
    state.xy+=uForce*g;
    state.z=min(1.,state.z+g*min(1.,length(uForce)*.5+.08));
  }
  float speed=length(state.xy);
  if(speed>5.)state.xy*=5./speed;
  outColor=encodeFluid(state);
}`)(R)),h?O==null||O.release():O==null||O.dispose(),O=R?null:(function(o,c,u){const s={splat:c(xo),curl:c(yo),vorticity:c(wo),divergence:c(bo),fade:c(Eo),pressure:c(To),gradient:c(Mo),advect:c(Ro)},x=Object.values(s);let E=0,M=0,y=[],H=[],F=null,L=null;const k=U=>{const P=o.createTexture();o.bindTexture(o.TEXTURE_2D,P),o.texParameteri(o.TEXTURE_2D,o.TEXTURE_MIN_FILTER,U),o.texParameteri(o.TEXTURE_2D,o.TEXTURE_MAG_FILTER,U),o.texParameteri(o.TEXTURE_2D,o.TEXTURE_WRAP_S,o.CLAMP_TO_EDGE),o.texParameteri(o.TEXTURE_2D,o.TEXTURE_WRAP_T,o.CLAMP_TO_EDGE),o.texImage2D(o.TEXTURE_2D,0,o.RGBA16F,E,M,0,o.RGBA,o.HALF_FLOAT,null);const T=o.createFramebuffer();return o.bindFramebuffer(o.FRAMEBUFFER,T),o.framebufferTexture2D(o.FRAMEBUFFER,o.COLOR_ATTACHMENT0,o.TEXTURE_2D,P,0),o.checkFramebufferStatus(o.FRAMEBUFFER)!==o.FRAMEBUFFER_COMPLETE?(o.deleteFramebuffer(T),o.deleteTexture(P),null):(o.viewport(0,0,E,M),o.clearColor(0,0,0,1),o.clear(o.COLOR_BUFFER_BIT),{texture:P,buffer:T})},z=U=>{U&&(o.deleteTexture(U.texture),o.deleteFramebuffer(U.buffer))},K=()=>{[...y,...H,F,L].forEach(z),y=[],H=[],F=L=null},G=(U,P)=>{const T=U.locations;return o.useProgram(U.program),o.bindFramebuffer(o.FRAMEBUFFER,P.buffer),o.uniform2f(T.uTexel,1/E,1/M),T},N=(U,P,T,ie)=>{o.activeTexture(o.TEXTURE0+T),o.bindTexture(o.TEXTURE_2D,ie),o.uniform1i(U[P],T)},Y=()=>o.drawArrays(o.TRIANGLES,0,3),se=U=>{const P=U[0];U[0]=U[1],U[1]=P};return{resize(U,P){K(),E=U,M=P;const T=[k(o.LINEAR),k(o.LINEAR),k(o.NEAREST),k(o.NEAREST),k(o.NEAREST),k(o.NEAREST)];return o.bindFramebuffer(o.FRAMEBUFFER,null),T.some(ie=>!ie)?(T.forEach(z),!1):(y=[T[0],T[1]],H=[T[2],T[3]],F=T[4],L=T[5],!0)},ready:()=>x.every(u),step(U,P){if(!y.length||!F||!L)return;let T;o.viewport(0,0,E,M),U&&(T=G(s.splat,y[1]),N(T,"uVelocity",0,y[0].texture),o.uniform2f(T.uFrom,U.from[0],U.from[1]),o.uniform2f(T.uTo,U.to[0],U.to[1]),o.uniform2f(T.uForce,U.force[0],U.force[1]),o.uniform1f(T.uAspect,P.aspect),o.uniform1f(T.uRadius,P.radius),Y(),se(y)),P.curl>0&&(T=G(s.curl,L),N(T,"uVelocity",0,y[0].texture),Y(),T=G(s.vorticity,y[1]),N(T,"uVelocity",0,y[0].texture),N(T,"uCurl",1,L.texture),o.uniform1f(T.uStrength,P.curl),o.uniform1f(T.uDt,P.dt),Y(),se(y)),T=G(s.divergence,F),N(T,"uVelocity",0,y[0].texture),Y(),T=G(s.fade,H[1]),N(T,"uSource",0,H[0].texture),o.uniform1f(T.uAmount,.8),Y(),se(H);for(let ie=0;ie<20;ie++)T=G(s.pressure,H[1]),N(T,"uPressure",0,H[0].texture),N(T,"uDivergence",1,F.texture),Y(),se(H);T=G(s.gradient,y[1]),N(T,"uPressure",0,H[0].texture),N(T,"uVelocity",1,y[0].texture),Y(),se(y),T=G(s.advect,y[1]),N(T,"uVelocity",0,y[0].texture),o.uniform1f(T.uDt,P.dt),o.uniform1f(T.uDissipation,P.dissipation),Y(),se(y),o.bindFramebuffer(o.FRAMEBUFFER,null)},get velocity(){var U;return((U=y[0])==null?void 0:U.texture)??null},get size(){return[E,M]},release:K,dispose(){K(),x.forEach(({program:U})=>o.deleteProgram(U))}}})(t,Et,o=>Tt(o))}const Yn=()=>{Zt();const o=a.fluid;if(!(o&&Jt()&&g&&w))return;_||Wn();const c=o.resolution,u=g/w;if(j=u>=1?c:Math.max(8,Math.round(c*u)),ue=u>=1?Math.max(8,Math.round(c/u)):c,W=!!O&&O.resize(j,ue),W)return void(oe=0);const s=()=>{Zt();for(let x=0;x<2;x++){const E=Ye(t.LINEAR);t.texImage2D(t.TEXTURE_2D,0,R?t.RGBA8:t.RGBA16F,j,ue,0,t.RGBA,R?t.UNSIGNED_BYTE:t.HALF_FLOAT,null);const M=t.createFramebuffer();t.bindFramebuffer(t.FRAMEBUFFER,M),t.framebufferTexture2D(t.FRAMEBUFFER,t.COLOR_ATTACHMENT0,t.TEXTURE_2D,E,0),re.push(E),ae.push(M)}return t.checkFramebufferStatus(t.FRAMEBUFFER)===t.FRAMEBUFFER_COMPLETE};if(!s()){if(R)throw new Error("The glyph field could not allocate its fluid buffers.");R=!0,Kn(),Wn(),s()}for(const x of ae)t.bindFramebuffer(t.FRAMEBUFFER,x),t.viewport(0,0,j,ue),t.clearColor(R?.5:0,R?.5:0,0,1),t.clear(t.COLOR_BUFFER_BIT);t.bindFramebuffer(t.FRAMEBUFFER,null),oe=0};function Kn(){A&&!h&&t.deleteProgram(A.program),X&&!h&&t.deleteProgram(X.program),Qt(),A=Et(Br(R)),X=Et(qr),(h?f.canCompileInParallel:V)||Kt(!0)}const Qn=()=>qn(a,q),en=()=>{Q=!0,Z()},Jn=()=>oo(a,q),tn=()=>{if(ee=Ne(a.charset).length,q=qt(a.cellSize,a.cellAspect,C,ee),ne=C,h){const u=Jn();if((D==null?void 0:D.key)!==u){const s=D;D=f.atlas(u,nn,en),f.releaseAtlas(s,en),B=D.texture}return void(Q=!0)}const{image:c}=nn();t.bindTexture(t.TEXTURE_2D,B),t.pixelStorei(t.UNPACK_PREMULTIPLY_ALPHA_WEBGL,!1),t.texImage2D(t.TEXTURE_2D,0,t.RGBA8,t.RGBA,t.UNSIGNED_BYTE,c),c.width=c.height=1,Q=!0};function nn(){return io(a,q,h)}const Mt=typeof document>"u"?void 0:document.fonts,rn=()=>{if(!(Mt!=null&&Mt.load))return;const o=Qn();Mt.load(o,Ne(a.charset).join("")).then(()=>{if(!Se&&!Fe&&Qn()===o)return h?void(D&&!D.final&&D.key===Jn()&&f.paintAtlas(D,nn)):(tn(),void Z())},()=>{})},Zn=o=>o.trim().replace(/^["']|["']$/g,"").toLowerCase(),er=()=>{if(st=[0,1],S&&a.levels!=="none")try{const o=document.createElement("canvas");o.width=o.height=64;const c=o.getContext("2d",{willReadFrequently:!0});S===je&&(c.imageSmoothingQuality="medium"),c.drawImage(S,0,0,64,64),st=to(c.getImageData(0,0,64,64).data)}catch{st=[0,1]}};let Ge=null,Rt=null,kt=!1,on=-1/0,At=new Uint8Array(0),Xe=null;const tr=()=>{if(kt=!1,!S||a.scope==="all")return;const o=Ee||1,c=ke||1,u=Math.max(8,Math.round(o>=c?96:96*o/c)),s=Math.max(8,Math.round(c>=o?96:96*c/o));try{if(Xe??(Xe=document.createElement("canvas").getContext("2d",{willReadFrequently:!0})),!Xe)return;const x=Xe.canvas;x.width!==u||x.height!==s?(x.width=u,x.height=s):Xe.clearRect(0,0,u,s),Xe.drawImage(S,0,0,u,s),Rt=(function(E,M,y,H,F){const L=M*y,k=F&&F.width===M&&F.height===y?F:{width:M,height:y,subject:new Float32Array(L),depth:new Float32Array(L),key:"none"},{subject:z,depth:K}=k;let G=0;for(let P=0;P<L;P++)E[4*P+3]<250&&G++;const N=P=>(.2126*E[4*P]+.7152*E[4*P+1]+.0722*E[4*P+2])/255;let Y="none",se=0;if(G>.01*L)Y="alpha";else{let P=0,T=0,ie=0;const ve=Math.max(1,Math.round(.03*Math.min(M,y)));for(let Me=0;Me<y;Me++)for(let ft=0;ft<M;ft++){if(ft>=ve&&Me>=ve&&ft<M-ve&&Me<y-ve)continue;const pn=N(Me*M+ft);P+=pn,T+=pn*pn,ie++}se=P/Math.max(1,ie),Math.sqrt(Math.max(0,T/Math.max(1,ie)-se*se))<.07&&(Y="ground")}for(let P=0;P<L;P++){const T=E[4*P+3]/255,ie=N(P);let ve=1;if(Y==="alpha")ve=T;else if(Y==="ground"){const Me=Math.max(0,Math.min(1,(Math.abs(ie-se)-.035)/.12));ve=Me*Me*(3-2*Me)}z[P]=1*ve,K[P]=ie*T}k.key=Y,wn.length<L&&(wn=new Float32Array(L));const U=wn;return Sr(z,U,M,y,Math.max(1,Math.round(.02*Math.min(M,y)))),Sr(K,U,M,y,Math.max(1,Math.round(.035*Math.min(M,y)))),k})(Xe.getImageData(0,0,u,s).data,u,s,0,Rt)}catch{return}if(Rt.key!=="none"){At.length!==u*s*4&&(At=new Uint8Array(u*s*4));for(let x=0;x<u*s;x++)At[4*x+3]=Math.round(255*Rt.subject[x]);Ge??(Ge=Ye(t.LINEAR)),t.bindTexture(t.TEXTURE_2D,Ge),t.pixelStorei(t.UNPACK_FLIP_Y_WEBGL,!1),t.pixelStorei(t.UNPACK_PREMULTIPLY_ALPHA_WEBGL,!1),t.texImage2D(t.TEXTURE_2D,0,t.RGBA8,u,s,0,t.RGBA,t.UNSIGNED_BYTE,At),kt=!0}},an=()=>{if(!S)return;const o=performance.now();S instanceof HTMLVideoElement&&!(o-On>1500)||(er(),On=o),S instanceof HTMLVideoElement&&!(o-on>160)||(tr(),on=o),t.bindTexture(t.TEXTURE_2D,d),t.pixelStorei(t.UNPACK_FLIP_Y_WEBGL,!1),t.pixelStorei(t.UNPACK_PREMULTIPLY_ALPHA_WEBGL,!1);try{t.texImage2D(t.TEXTURE_2D,0,t.RGBA8,t.RGBA,t.UNSIGNED_BYTE,S),t.generateMipmap(t.TEXTURE_2D);const c=S;Ht=(c.videoWidth||c.naturalWidth||c.width)*(c.videoHeight||c.naturalHeight||c.height)*4*4/3}catch(c){ht(c)}},nr=()=>{R=!t.getExtension("EXT_color_buffer_float")&&!t.getExtension("EXT_color_buffer_half_float"),Kn(),we=h?f.vertexArray():t.createVertexArray(),h||(B=Ye(t.NEAREST)),d=Ye(t.LINEAR_MIPMAP_LINEAR),t.bindTexture(t.TEXTURE_2D,d),t.texImage2D(t.TEXTURE_2D,0,t.RGBA8,1,1,0,t.RGBA,t.UNSIGNED_BYTE,new Uint8Array([0,0,0,255])),t.generateMipmap(t.TEXTURE_2D),g=w=0,ne=0,Ke(),rn(),S&&an()};function Ke(){const o=e.clientWidth||e.clientHeight?{width:e.clientWidth,height:e.clientHeight}:e.getBoundingClientRect();zn=!o.width||!o.height;const c=ao(a,o.width,o.height,h?f.scale:1),u=Math.max(1,Math.round(o.width*c)),s=Math.max(1,Math.round(o.height*c));if(u===g&&s===w&&c===C)return!1;const x=!g||Math.abs(u/s-g/w)>.01;return C=c,g=u,w=s,e.width=g,e.height=w,ne!==C&&tn(),(x||Jt()&&!re.length&&!W)&&Yn(),Q=!0,!0}let rr=null,or=[0,0,0],ar=[1,1,1],ir=[0,0,0,0],Qe=Fn(1,1,1,1,"cover"),lr=null,sr=0,cr=0,ur=0,hr=0;const fr=()=>{var z,K;rr!==a&&(rr=a,or=En(a.ink),ar=En(a.highlight==="auto"?"#ffffff":a.highlight),ir=a.background==="transparent"?[0,0,0,0]:[...En(a.background),1]),lr===a&&sr===g&&cr===w&&ur===Ee&&hr===ke||(lr=a,sr=g,cr=w,ur=Ee,hr=ke,Qe=Fn(g,w,Ee||1,ke||1,a.fit,a.focus)),h&&f.begin(g,w);const o=a.fluid,c=S&&it?q.height/w*it*Qe.scale[1]:1,u=(()=>{const[G,N]=$n(),Y=Math.ceil(g/G-1e-4),se=Math.ceil(w/N-1e-4);if($&&$.columns===Y&&$.rows===se)return $;Qt();const U=()=>{const ve=Ye(t.NEAREST);return t.texStorage2D(t.TEXTURE_2D,1,R?t.RGBA8:t.RGBA16F,Y,se),ve},P=U(),T=U(),ie=t.createFramebuffer();return t.bindFramebuffer(t.FRAMEBUFFER,ie),t.framebufferTexture2D(t.FRAMEBUFFER,t.COLOR_ATTACHMENT0,t.TEXTURE_2D,P,0),t.framebufferTexture2D(t.FRAMEBUFFER,t.COLOR_ATTACHMENT1,t.TEXTURE_2D,T,0),t.drawBuffers([t.COLOR_ATTACHMENT0,t.COLOR_ATTACHMENT1]),$={ink:P,under:T,buffer:ie,columns:Y,rows:se}})(),s=A.locations,x=X.locations,E=et(a),M=!We();t.bindFramebuffer(t.FRAMEBUFFER,u.buffer),t.viewport(0,0,u.columns,u.rows),t.useProgram(A.program),t.activeTexture(t.TEXTURE0),t.bindTexture(t.TEXTURE_2D,d),t.uniform1i(s.uSource,0),t.activeTexture(t.TEXTURE2),t.bindTexture(t.TEXTURE_2D,(W?O==null?void 0:O.velocity:re[oe])??d),t.uniform1i(s.uFluid,2),t.activeTexture(t.TEXTURE3),t.bindTexture(t.TEXTURE_2D,De??d),t.uniform1i(s.uTrailMap,3),t.activeTexture(t.TEXTURE6),t.bindTexture(t.TEXTURE_2D,Ge??d),t.uniform1i(s.uScope,6),t.uniform1f(s.uScopeMode,kt&&a.scope!=="all"?a.scope==="subject"?1:2:0),t.uniform1f(s.uFluidMode,W?1:0),t.uniform1i(s.uHoverMode,E==="fluid"?1:E==="lantern"?2:E==="trail"&&bt?3:E==="loupe"?4:E==="reveal"?5:0),t.uniform1f(s.uTrailAmount,a.trail.strength),t.uniform3f(s.uLight,((z=J.at)==null?void 0:z[0])??-1,((K=J.at)==null?void 0:K[1])??-1,jt(E)?J.strength:0),t.uniform3fv(s.uTrail,ut),t.uniform1f(s.uLightRadius,a.lantern.radius),t.uniform1f(s.uLightGain,a.lantern.strength),t.uniform1f(s.uClock,performance.now()/1e3),t.uniform1f(s.uFlicker,M?1:0),t.uniform2f(s.uFluidTexel,1/Math.max(1,j),1/Math.max(1,ue)),t.uniform2f(s.uResolution,g,w);const[y,H]=$n();var F;t.uniform2f(s.uCell,y,H),t.uniform1f(s.uCount,ee),t.uniform1f(s.uBlend,a.blend==="on"||ee<=9&&(a.blend==="coarse"||a.blend==="auto"&&me)?1:0),t.uniform4f(s.uSourceMap,Qe.scale[0],Qe.scale[1],Qe.offset[0],Qe.offset[1]),t.uniform1i(s.uScene,S?0:(F=a.scene,vo[F]??3)),t.uniform1i(s.uMark,It.indexOf(a.mark)),t.uniform1f(s.uOrder,a.glyphOrder==="text"?1:0),t.uniform1f(s.uScale,a.scale);const L=a.clear;t.uniform4f(s.uClear,L?L.x:0,L?L.y:0,L?L.width:0,L?L.height:0),t.uniform1f(s.uClearFeather,L?L.feather:.1),t.uniform1f(s.uClearShape,L&&L.shape==="rect"?1:0);const k=performance.now();$e.fill(0);for(let G=0;G<Te.length;G++){const N=Te[G],Y=4*G;$e[Y]=N.x,$e[Y+1]=N.y,$e[Y+2]=(k-N.born)/1e3,$e[Y+3]=N.strength}t.uniform4fv(s.uPulse,$e),t.uniform3fv(s.uInk,or),t.uniform3fv(s.uHighlight,ar),t.uniform1f(s.uHighlightAuto,a.highlight==="auto"?1:0),t.uniform4fv(s.uBackground,ir),t.uniform1f(s.uBrightness,a.brightness),t.uniform1f(s.uContrast,a.contrast),t.uniform1f(s.uGamma,a.gamma),t.uniform1f(s.uInvert,a.invert?1:0),t.uniform1f(s.uVignette,a.vignette),t.uniform1f(s.uSourceColor,a.color==="source"?1:0),t.uniform1f(s.uRefraction,o?o.refraction*(M?1:.25):0),t.uniform1f(s.uGlow,o?o.glow:0),t.uniform1f(s.uTime,Nn),t.uniform1f(s.uDrift,M?a.drift:0),t.uniform1f(s.uFluidOn,o&&E==="fluid"&&(re.length||W)?1:0),t.uniform1f(s.uLod,Math.max(0,Math.log2(Math.max(1,c))-.5)),t.uniform1f(s.uSaturation,a.saturation),t.uniform1f(s.uShade,a.shade),t.uniform1f(s.uBackdrop,S?a.backdrop:0),t.uniform2f(s.uLevels,S&&a.levels==="auto"?st[0]:0,S&&a.levels==="auto"?st[1]:1),t.bindVertexArray(we),t.drawArrays(t.TRIANGLES,0,3),t.bindFramebuffer(t.FRAMEBUFFER,null),t.viewport(0,0,g,w),t.useProgram(X.program),t.activeTexture(t.TEXTURE1),t.bindTexture(t.TEXTURE_2D,B),t.uniform1i(x.uAtlas,1),t.activeTexture(t.TEXTURE4),t.bindTexture(t.TEXTURE_2D,u.ink),t.uniform1i(x.uCells,4),t.activeTexture(t.TEXTURE5),t.bindTexture(t.TEXTURE_2D,u.under),t.uniform1i(x.uUnder,5),t.uniform2f(x.uResolution,g,w),t.uniform2f(x.uCell,q.width,q.height),t.uniform2f(x.uPitch,y,H),t.uniform1f(x.uColumns,q.columns),t.uniform1f(x.uDotScale,a.dotScale),t.uniform2f(x.uDir,Math.cos(a.markAngle*Math.PI/180),Math.sin(a.markAngle*Math.PI/180)),t.uniform1i(x.uMark,It.indexOf(a.mark)),t.uniform1f(x.uSourceColor,a.color==="source"?1:0),t.drawArrays(t.TRIANGLES,0,3),h&&f.present(p,g,w)},dr=()=>!Se&&!Fe&&!xt&&!zn&&(wt||qe)&&!(typeof document<"u"&&document.hidden),St=o=>{if(Ae=!1,!dr())return;const c=(o-zt)/1e3,u=Math.min(1/30,Math.max(1/480,c));zt=o,Nt=.9*Nt+.1*Math.min(c,.25),ct=Nt>1/24?ct+1:0;try{h&&f.overloaded&&Be<2&&ct>10&&(Be=2),ct>45&&Be<2&&(!h||f.atMinimum)&&(ct=0,Be++,Be===1&&C>1?(a={...a,maxPixelRatio:1},Ke()):Be=2);const s=Kt(),x=!(yt||Be>=2&&!(S instanceof HTMLVideoElement)||(!(S instanceof HTMLVideoElement)||S.paused||S.ended)&&(We()||!(!S&&a.scene!=="none"||a.drift>0)));x&&(Nn+=u*a.speed),S instanceof HTMLVideoElement&&S.readyState>=2&&vt&&(vt=!("requestVideoFrameCallback"in S),an(),Q=!0);const E=et(a);let M=E==="fluid"&&(_e!==null||o<$t);if(M?((y,H)=>{const F=a.fluid;if(!F||!ae.length&&!W||(W?!(O!=null&&O.ready()):!_||!Tt(_)))return;const L=g/w,k=_e;if(_e=null,W&&O){const N=5200*F.force*(We()?.35:1);return k&&($t=H+1e3*Math.min(9,6/Math.max(.3,F.dissipation))),void O.step(k?{from:k.from,to:k.to,force:[(k.to[0]-k.from[0])*N,(k.to[1]-k.from[1])*N]}:null,{dt:Math.min(y,1/60),radius:1.1*F.radius,curl:F.curl,dissipation:F.dissipation,aspect:L})}let z=0,K=0;if(k){const N=(We()?1:3)*F.force;z=(k.to[0]-k.from[0])*L*N*4,K=(k.to[1]-k.from[1])*N*4,$t=H+7/Math.min(F.dissipation,1.1*F.dissipation)*1e3}const G=_.locations;t.bindFramebuffer(t.FRAMEBUFFER,ae[1-oe]),t.viewport(0,0,j,ue),t.useProgram(_.program),t.activeTexture(t.TEXTURE0),t.bindTexture(t.TEXTURE_2D,re[oe]),t.uniform1i(G.uPrevious,0),t.uniform2f(G.uSize,j,ue),t.uniform1f(G.uAspect,L),t.uniform1f(G.uDt,y),t.uniform1f(G.uDecay,F.dissipation),t.uniform1f(G.uEnergyDecay,1.1*F.dissipation),t.uniform1f(G.uRadius,F.radius),t.uniform1f(G.uSplat,k?1:0),t.uniform2f(G.uFrom,(k==null?void 0:k.from[0])??0,(k==null?void 0:k.from[1])??0),t.uniform2f(G.uTo,(k==null?void 0:k.to[0])??0,(k==null?void 0:k.to[1])??0),t.uniform2f(G.uForce,z,K),t.drawArrays(t.TRIANGLES,0,3),oe=1-oe})(u,o):_e=null,jt(E)&&(M=((y,H)=>{const F=a.lantern,L=J.target;if(L){const z=J.at;if(z)if(We())z[0]=L[0],z[1]=L[1];else{const N=1-Math.exp(-F.follow*y);z[0]+=(L[0]-z[0])*N,z[1]+=(L[1]-z[1])*N}else J.at=[L[0],L[1]];const K=J.at,G=J.trail[J.trail.length-1];F.trail>0&&(!G||H-G.born>28&&Math.hypot(K[0]-G.x,K[1]-G.y)>.004)&&(J.trail.push({x:K[0],y:K[1],born:H}),J.trail.length>12&&J.trail.shift())}J.strength+=((L?1:0)-J.strength)*(1-Math.exp(-y*(L?9:5)));const k=J.trail;for(let z=k.length-1;z>=0;z--)F.trail>0&&H-k[z].born<4e3*F.trail||k.splice(z,1);ut.fill(0);for(let z=0;z<k.length;z++){const K=k[z];ut[3*z]=K.x,ut[3*z+1]=K.y,ut[3*z+2]=.55*Math.exp(-(H-K.born)/1e3/Math.max(.01,F.trail))}return J.strength>.003||k.length>0})(u,o)||M),E==="trail"){const y=!!(te!=null&&te.active);M=(H=>{const F=Vn();return!(!F||!F.active||(F.step(H),t.bindTexture(t.TEXTURE_2D,De),t.texSubImage2D(t.TEXTURE_2D,0,0,0,F.width,F.height,t.RGBA,t.UNSIGNED_BYTE,F.pixels),bt=!0,0))})(u)||M,y&&!(te!=null&&te.active)&&(Q=!0)}if(Te.length){for(let y=Te.length-1;y>=0;y--)o-Te[y].born>=2400&&Te.splice(y,1);M=!0,Te.length||(Q=!0)}s&&(M||Q||x&&o-Hn>=mo*Gr())&&(fr(),Hn=o,Q=!1,!qe||at&&!S||(qe=!1,Yt())),(x||M||_e||!s)&&(Ae=!0,An(St))}catch(s){xt=!0,ht(s)}};function Z(){!Ae&&dr()&&(zt=performance.now(),Ae=!0,An(St))}const mr=(o,c)=>{const u=e.getBoundingClientRect();if(!u.width||!u.height)return null;const s=(o-u.left)/u.width,x=(c-u.top)/u.height;return s<-.05||s>1.05||x<-.05||x>1.05?null:[s,1-x]},ln=o=>{const c=et(a);if(c==="none"||yt)return Le=null,void(J.target=null);if(jt(c))return J.target=o?[o[0],1-o[1]]:null,Le=null,void Z();if(c==="trail"){const u=Vn();return u?(o?u.move(o[0],1-o[1],a.trail.radius,performance.now()):u.leave(),Le=null,void Z()):void 0}o?(Le&&(_e=_e?{from:_e.from,to:o}:{from:Le,to:o}),Le=o,Z()):Le=null},gr=n.interactionTarget??window,pr={target:gr,move:o=>{const c=mr(o.clientX,o.clientY);gr===window&&(pr.inside=!!c),ln(c)},leave:()=>{Le=null,te==null||te.leave(),J.target&&(J.target=null,Z())},near:()=>wt},Je=[Or(pr)],sn=()=>{const o=Ke();Vt(),(o||Q)&&Z()};let cn=i?yn(i.element,sn):()=>{};Je.push(yn(e,sn),()=>cn()),Je.push((function(o,c,u){if(typeof IntersectionObserver!="function")return()=>{};let s=xn.get(c);if(!s){const M=new Map;s={observer:new IntersectionObserver(Nr(M),{rootMargin:c}),watch:M},xn.set(c,s)}const{observer:x,watch:E}=s;return Vr(E,o,u)||x.unobserve(o),x.observe(o),()=>{jr(E,o,u)&&x.unobserve(o),E.size||(x.disconnect(),xn.delete(c))}})(e,"25% 0px",o=>{wt=o.isIntersecting,wt&&(Q=!0,Z())})),Je.push(Hr(()=>{document.hidden||(Q=!0,Z())})),Je.push(zr(()=>{Q=!0,Z()})),Je.push(ot(In,o=>{const c=o.fontfaces??[],u=a.font.split(",").map(Zn);c.some(s=>u.includes(Zn(s.family)))&&rn()}));const un=o=>{o==null||o.preventDefault(),Fe=!0,Ae&&Sn(St),Ae=!1},hn=()=>{Fe=!1,h&&(t=f.gl),fe.length=0,pe=null,D=null,A=X=_=null,$=null,re=[],ae=[],O=null,W=!1,De=null,te=null,bt=!1,Ge=null,kt=!1;try{nr(),Q=!0,Z()}catch(o){ht(o)}},vr={lost:()=>un(),restored:hn,rescale:()=>{Ke()&&Z()},size:()=>[g,w],bytes:()=>{const o=R?4:8;return g*w*4+Ht+($?$.columns*$.rows*o*2:0)+j*ue*o*(W?6:re.length)+(te?te.width*te.height*4:0)}};h?f.join(vr):(e.addEventListener("webglcontextlost",un),e.addEventListener("webglcontextrestored",hn));const fn=()=>{be&&(be.pause(),be.removeAttribute("src"),be.load(),be=null),lt&&S instanceof HTMLVideoElement&&"cancelVideoFrameCallback"in S&&S.cancelVideoFrameCallback(lt),lt=0},dn=()=>{je==null||je.close(),je=null},po=async o=>{const c=await createImageBitmap(o,{premultiplyAlpha:"none"}),u=[c.width,c.height],s=(function(x,E,M,y,H,F,L=8){if(!(M>=16&&y>=16&&x>0&&E>0&&H>0))return 1;const k=L/H/(F==="cover"?Math.min(x/M,E/y):Math.max(x/M,E/y));return k>.5?1:2**-Math.floor(Math.log2(1/k))})(c.width,c.height,g,w,q.height,a.fit);if(s===1)return{element:c,owned:!0,natural:u};try{return{element:await createImageBitmap(c,{resizeWidth:Math.max(1,Math.round(c.width*s)),resizeHeight:Math.max(1,Math.round(c.height*s)),resizeQuality:"medium",premultiplyAlpha:"none"}),owned:!0,natural:u}}finally{c.close()}},mn=o=>new Promise((c,u)=>{const s=new Image;s.crossOrigin="anonymous",s.decoding="async",s.onload=()=>c({element:s,owned:!1}),s.onerror=()=>u(new Error("The glyph field could not load this image.")),s.src=o}),xr=(o,c,u)=>{var M,y;const{element:s,owned:x}=o;if(Se||u!==at)return void(x&&s instanceof HTMLVideoElement?(s.pause(),s.removeAttribute("src"),s.load()):x&&s.close());s!==S&&(fn(),dn()),be=x&&s instanceof HTMLVideoElement?s:null,!x||s instanceof HTMLVideoElement||(je=s);const E=s!==S;S=s,me=typeof c=="string"||c instanceof HTMLImageElement||c instanceof HTMLVideoElement,it=s instanceof HTMLVideoElement?s.videoHeight:s instanceof HTMLImageElement?s.naturalHeight:s.height,Ee=((M=o.natural)==null?void 0:M[0])??(s instanceof HTMLVideoElement?s.videoWidth:s instanceof HTMLImageElement?s.naturalWidth:s.width),ke=((y=o.natural)==null?void 0:y[1])??it,vt=!0,s instanceof HTMLVideoElement&&E&&(H=>{const F=H;if(!F.requestVideoFrameCallback)return;const L=()=>{Se||S!==H||(vt=!0,Z(),lt=F.requestVideoFrameCallback(L))};lt=F.requestVideoFrameCallback(L)})(s),Fe||an(),Q=!0,Z()},yr=(o,c)=>{qe&&o===at&&(qe=!1,Yt()),ht(c)},gn=o=>{const c=++at;if(!o)return fn(),dn(),S=null,Ee=ke=it=Ht=0,me=!1,Q=!0,Z(),Promise.resolve();if((u=>!(typeof u=="string"||u instanceof HTMLImageElement&&!u.complete||u instanceof HTMLVideoElement&&u.readyState<2))(o)){try{xr({element:o,owned:!1},o,c)}catch(u){yr(c,u)}return Promise.resolve()}return(async u=>{if(typeof u!="string")return u instanceof HTMLImageElement&&!u.complete?u.decode().then(()=>({element:u,owned:!1})):u instanceof HTMLVideoElement&&u.readyState<2?new Promise((E,M)=>{u.addEventListener("loadeddata",()=>E({element:u,owned:!1}),{once:!0}),u.addEventListener("error",()=>M(new Error("The glyph field could not load this video.")),{once:!0})}):{element:u,owned:!1};if(Uo.test(u))return new Promise((E,M)=>{const y=document.createElement("video");y.crossOrigin="anonymous",y.muted=!0,y.loop=!0,y.playsInline=!0,y.preload="auto",y.addEventListener("loadeddata",()=>{y.play().catch(()=>{}),E({element:y,owned:!0})},{once:!0}),y.addEventListener("error",()=>M(new Error("The glyph field could not load this video.")),{once:!0}),y.src=u});if(typeof createImageBitmap!="function"||typeof fetch!="function")return mn(u);let s;try{s=await(await fetch(u).then(E=>{if(!E.ok)throw new Error("The glyph field could not load this image.");return E})).blob()}catch(E){if(E instanceof Error&&E.message.startsWith("The glyph field"))throw E;return mn(u)}if(!/svg/i.test(s.type))try{return await po(s)}catch{}const x=URL.createObjectURL(s);try{return await mn(x)}finally{URL.revokeObjectURL(x)}})(o).then(u=>xr(u,o,c),u=>yr(c,u))};nr(),Vt();const wr=n.source??null;return wr?(at=1,gn(wr)):Z(),{canvas:e,ready:go,get settings(){return a},update(o){const c=a,{element:u,rest:s}=l(o);"clear"in o&&(i==null?void 0:i.element)!==(u==null?void 0:u.element)?(cn(),i=u,cn=u?yn(u.element,sn):()=>{}):"clear"in o&&(i=u),a=Bt(u?{...s,clear:a.clear}:s,a),Vt(),o.label&&e.setAttribute("aria-label",o.label),a.maxPixelRatio!==c.maxPixelRatio&&Ke(),a.charset===c.charset&&a.cellSize===c.cellSize&&a.cellAspect===c.cellAspect&&a.font===c.font&&a.fontWeight===c.fontWeight&&a.glyphScale===c.glyphScale||(tn(),rn()),a.levels!==c.levels&&er(),a.scope===c.scope||Fe||(tr(),on=performance.now()),((!!c.fluid&&et(c)==="fluid")!==Jt()||a.fluid&&c.fluid&&a.fluid.resolution!==c.fluid.resolution)&&Yn(),"source"in o&&gn(o.source??null),Q=!0,Z()},setSource:gn,pointer(o,c=0){if(o===null)return void ln(null);const u=e.getBoundingClientRect();ln(mr(u.left+o,u.top+c))},redraw(){if(!(Se||Fe||xt)&&g)try{Kt(!(qe&&(h?f.canCompileInParallel:V)))&&(fr(),qe||(Q=!1))}catch(o){xt=!0,ht(o)}},pulse(o={}){if(Se||We())return;const c=e.getBoundingClientRect();if(!c.width||!c.height)return;let u=o.x??c.width/2,s=o.y??c.height/2;if(o.element){const E=o.element.getBoundingClientRect();u=E.left+E.width/2-c.left,s=E.top+E.height/2-c.top}const x=Math.max(0,Math.min(2,o.strength??1));Te=[...Te,{x:u/c.width,y:s/c.height,born:performance.now(),strength:x}].slice(-4),Z()},play(){yt=!1,S instanceof HTMLVideoElement&&S.play().catch(()=>{}),Q=!0,Z()},pause(){yt=!0,S instanceof HTMLVideoElement&&S.pause()},resize(){Ke()&&Z()},destroy(){if(!Se){if(Se=!0,Ae&&Sn(St),Ae=!1,Je.forEach(o=>o()),h?f.leave(vr):(e.removeEventListener("webglcontextlost",un),e.removeEventListener("webglcontextrestored",hn)),fn(),dn(),S=null,!Fe)if(Zt(),Qt(),t.deleteTexture(d),De&&t.deleteTexture(De),Ge&&t.deleteTexture(Ge),h)f.releaseAtlas(D,en),O==null||O.release();else{t.deleteTexture(B);for(const o of[A,X,_])o&&t.deleteProgram(o.program);O==null||O.dispose(),fe.forEach(o=>t.deleteShader(o)),we&&t.deleteVertexArray(we)}D=null}}}}const Yo=Object.freeze(Object.defineProperty({__proto__:null,DEFAULT_FIELD_FLUID:ye,DEFAULT_FIELD_LANTERN:Jr,DEFAULT_FIELD_SETTINGS:eo,DEFAULT_FIELD_TRAIL:Zr,FIELD_HOVERS:Qr,FIELD_MARKS:It,FIELD_SCENES:Kr,activeFieldHover:et,ambientFactor:Gr,cancelFieldFrame:Sn,currentFieldContext:$r,fieldCellLayout:qt,fieldClearFromRects:no,fieldGlyphs:Ne,fieldLevels:to,fieldSourceRect:Fn,glyphFieldTier:ro,isGlyphFieldSupported:Io,isSharedFieldCanvas:Yr,mountGlyphField:so,normalizeFieldSettings:Bt,prewarmGlyphAtlas:lo,requestFieldFrame:An,watchMotionPreference:zr,watchPointer:Or,watchTheme:_o,watchVisibility:Hr},Symbol.toStringTag,{value:"Module"}));var co={ascii:{mark:"glyph",charset:" .:-=+*#%@"},blocks:{mark:"glyph",charset:" ░▒▓█"},braille:{mark:"glyph",charset:" ⠁⠃⠇⡇⣇⣧⣷⣿"},dots:{mark:"dot"},lines:{mark:"glyph",charset:" -═╪╬"},cross:{mark:"glyph",charset:" ·+xX#"},diagonal:{mark:"glyph",charset:" /╱╳"},diamond:{mark:"glyph",charset:" ·◇◈◆"},mixed:{mark:"glyph",charset:" .:+/◇#@"},pixel:{mark:"pixel"},mosaic:{mark:"mosaic"},lego:{mark:"lego"},voxel:{mark:"voxel"},disco:{mark:"disco"},dither:{mark:"dither"},hex:{mark:"hex"},led:{mark:"led"},cmyk:{mark:"cmyk"},stipple:{mark:"stipple"},engraving:{mark:"engraving"},crosshatch:{mark:"crosshatch"},woodblock:{mark:"woodblock"},risograph:{mark:"risograph"},pointillism:{mark:"pointillism"}},Bo=e=>typeof e=="string"&&Object.prototype.hasOwnProperty.call(co,e),qo=new RegExp("\\p{L}","u"),He=null,Ln=new Set,_n=new Set;function Ko(e){const{charset:n,style:r}=e??{};He=e?{...n?{charset:n}:{},...Bo(r)?{style:r}:{}}:null;for(const l of _n)l.relook();for(const l of[...Ln])l(He)}var Qo=()=>He;function Jo(e){return Ln.add(e),()=>{Ln.delete(e)}}var Gt=12,le=new Set,ce=null,ge=null,Dt=0,gt=-1e9,Pt=0,Ct=0,Tn=!1,Mn=!1,Go=()=>{var n;const e=(n=ge==null?void 0:ge.timeline)==null?void 0:n.currentTime;return typeof e=="number"?e:null};function Zo(e){Gt=Number.isFinite(e)&&e>=1?Math.floor(e):12,Ve()}var ea=()=>Gt,Rn=()=>typeof performance<"u"?performance.now():Date.now(),Gn=()=>typeof innerHeight=="number"?innerHeight:0;function Xt(e,n=e.getBoundingClientRect()){if(e.isConnected===!1||typeof e.getClientRects=="function"&&e.getClientRects().length===0)return 1/0;const r=n.bottom<0?-n.bottom:n.top-Gn();return r>0?r:0}function Xo(){var e;try{const n=(e=ge==null?void 0:ge.querySelectorAll)==null?void 0:e.call(ge,"dialog:modal");return n!=null&&n.length?n[n.length-1]:null}catch{return null}}var Oo=1e8;function uo(e,n){if(e.eager)return void(e.wanted=!0);const r=Gn(),l=Xt(e.host,n);!r||l<=2*r?e.wanted=!0:l>3*r&&(e.wanted=!1)}function Ho(e){let n=!1;for(const r of e)if(r!=null&&r.target)for(const l of le)l.host===r.target&&(uo(l,r.boundingClientRect??void 0),n=!0);n&&Ve()}var Dr=()=>Ve(),Ot=0,Ut=0,Pr=()=>{const e=typeof scrollY=="number"?scrollY:0;e!==Ut&&(Ot=e>Ut?1:-1),Ut=e};function zo(e,n){return!n||!Ot||e.getBoundingClientRect().top>0==Ot>0?n:2*n}var de=null,ho=e=>{var n,r,l,i;de!==e&&((n=de==null?void 0:de.removeEventListener)==null||n.call(de,"close",Dr,!0),(r=de==null?void 0:de.removeEventListener)==null||r.call(de,"scroll",Pr,!0),(l=e==null?void 0:e.addEventListener)==null||l.call(e,"close",Dr,!0),(i=e==null?void 0:e.addEventListener)==null||i.call(e,"scroll",Pr,{capture:!0,passive:!0}),de=e)},Dn=new WeakSet,nt=0;function Xn(){nt||typeof requestIdleCallback!="function"||(nt=requestIdleCallback(No,{timeout:2e3}))}function No(e){nt=0;const n=[...le].filter(r=>!r.live&&r.prewarm&&!Dn.has(r)).sort((r,l)=>Xt(r.host)-Xt(l.host));for(const r of n){if(e.timeRemaining()<2&&!e.didTimeout)return void Xn();Dn.add(r);try{r.prewarm()}catch{}}}function Vo(e){ge!==e.doc&&(ge=e.doc,le.clear(),Dt=Pt=Ct=Ot=0,gt=-1e9,Ut=typeof scrollY=="number"?scrollY:0),ho(ge),le.add(e);const n=(function(){const r=typeof IntersectionObserver=="function"?IntersectionObserver:null;if(!r)return null;if((ce==null?void 0:ce.ctor)!==r){ce==null||ce.list.forEach(l=>l.disconnect()),ce={ctor:r,list:["200% 0px","300% 0px","0px"].map(l=>new r(Ho,{rootMargin:l}))};for(const l of le)ce.list.forEach(i=>i.observe(l.host))}return ce.list})();n==null||n.forEach(r=>r.observe(e.host)),n?uo(e):e.wanted=!0,Ve(),e.live||Xn()}function Ve(e=!1){var n,r;if(Tn)Mn=!0;else{Tn=!0;try{do{Mn=!1;const l=Rn(),i=Go();(e||(i===null?l-gt>16||l<gt:i!==gt))&&(gt=i??l,Pt=Ct=0);for(const d of le)d.host.isConnected===!1&&(d.wanted=!1);for(const d of le)d.live&&!d.wanted&&d.release();const a=[...le].filter(d=>d.live&&!d.shared),m=Xo(),v=d=>!!m&&!m.contains(d.host),f=d=>Xt(d.host)+(v(d)?Oo:0),p=d=>d.eager&&!v(d),h=()=>a.filter(d=>d.live&&!d.shared).map(d=>[d,f(d)+(p(d)?0:1e9)]).sort((d,A)=>A[1]-d[1]);for(let d=a.length-Gt,A=d>0?h():[];d>0;d--)(n=A.shift())==null||n[0].release();const b=$r(),t=1*Gn(),R=()=>[...le].filter(d=>d.live&&d.shared&&f(d)>t).sort((d,A)=>f(A)-f(d));if(b!=null&&b.overBudget)for(const d of R()){if(!b.overBudget)break;d.release()}const V=[...le].filter(d=>d.wanted&&!d.live).map(d=>[d,d.eager?-1:f(d)]).map(([d,A])=>[d,A,A>0?zo(d.host,A):A]).sort((d,A)=>d[2]-A[2]);let B=0;for(const[d,A]of V){if(!le.has(d)||d.live)continue;if(b&&!b.roomy){const _=R()[0];if(!_||f(_)<=f(d)||(_.release(),!b.roomy))break}if(!d.eager&&(Pt>=4||Ct>=2)&&(!e||B)){Dt||(Dt=requestAnimationFrame(()=>{Dt=0,Ve(!0)}));break}if(a.filter(_=>_.live&&!_.shared).length>=Gt){const _=(r=h()[0])==null?void 0:r[0];if(!_||(p(_)?!d.eager||f(_)<=f(d):!d.eager&&f(_)<=A))break;_.release()}const X=Rn();d.mount(),Pt+=Rn()-X,B++,d.eager||Ct++,d.live&&!d.shared&&a.push(d)}}while(Mn)}finally{Tn=!1}}}var fo='ui-monospace, "SF Mono", Menlo, Consolas, monospace',Cr=new Map;function jo(e,n=fo){const r=`${n}\0${e}`,l=Cr.get(r);if(l)return l;let i={max:1,mean:1};try{const a=typeof document>"u"?null:Object.assign(document.createElement("canvas"),{width:48,height:48}).getContext("2d",{willReadFrequently:!0});if(a&&typeof a.getImageData=="function"){a.font=`500 36px ${n}`,a.textAlign="center",a.textBaseline="middle";const m=t=>{a.clearRect(0,0,48,48),a.fillStyle="#000",a.fillText(t,24,25);const R=a.getImageData(0,0,48,48).data;let V=0;for(let B=3;B<R.length;B+=4)V+=R[B];return V/255},v=t=>[...t].filter(R=>R.trim()).map(m),f=t=>t.reduce((R,V)=>R+V,0)/Math.max(1,t.length),p=v(e),h=v(".:-=+*#%@"),b=m("@");p.length&&b>0&&f(h)>0&&(i={max:Math.max(...p)/b,mean:f(p)/f(h)})}}catch{}return Cr.set(r,i),i}function $o(e,n){const{max:r,mean:l}=jo(e,n.font??fo),i=n.glyphScale??.8,a=n.gamma??1,m=n.fontWeight??500;return r>1.15?{glyphScale:Math.max(.3,i*Math.max(.62,r**-.36)),gamma:a,fontWeight:m}:l>1.5?{glyphScale:Math.max(.3,i*l**-.25),gamma:a,fontWeight:m}:r<.85?{glyphScale:Math.min(1.6,i*Math.min(1.3,r**-.3)),gamma:a,fontWeight:r<.6?Math.max(m,700):m}:{glyphScale:i,gamma:a,fontWeight:m}}var rt=new WeakMap;function ta(e,n){var r;for(const l of((r=rt.get(e))==null?void 0:r.keys())??[])l.getAttribute("src")!==n&&(l.src=n)}function na(e){const n=rt.get(e);rt.delete(e),n==null||n.forEach(r=>r())}function Wo(e,n,r,l,i){const a=document.createElement("img");a.alt="",a.decoding="async",a.setAttribute("aria-hidden","true"),a.setAttribute("style",`${r};object-fit:cover`),a.style.opacity="1",a.dataset.asciiPoster="",a.src=n;const m=e.style.getPropertyValue("position"),v=typeof getComputedStyle=="function"&&getComputedStyle(e).position==="static";v&&e.style.setProperty("position","relative");const f=e.style.getPropertyValue("isolation"),p=/z-index:\s*-1/.test(r)&&typeof getComputedStyle=="function"&&getComputedStyle(e).isolation==="auto";p&&e.style.setProperty("isolation","isolate"),(i==null?void 0:i.parentElement)===e?e.insertBefore(a,i):e.appendChild(a);let h=!1;const b=()=>{var R;h||(h=!0,a.remove(),(R=rt.get(l))==null||R.delete(a),v&&(m?e.style.setProperty("position",m):e.style.removeProperty("position")),p&&(f?e.style.setProperty("isolation",f):e.style.removeProperty("isolation")))},t=rt.get(l)??new Map;return t.set(a,b),rt.set(l,t),{show(R){a.style.opacity=R?"1":"0"},remove:b}}function Ur(e){var r;const n=document.createElement("canvas");for(const l of((r=e.getAttributeNames)==null?void 0:r.call(e))??[])l!=="width"&&l!=="height"&&n.setAttribute(l,e.getAttribute(l));return n.style.cssText=e.style.cssText,Object.assign(n.dataset,e.dataset),e.replaceWith?e.replaceWith(n):e.parentElement&&(e.parentElement.insertBefore(n,e),e.remove()),n}function ra(e){const{host:n,kit:r}=e;let l,i=e.canvas,a=null,m=!1,v=!1,f=!1,p=!1,h={},b={},t=e.source,R=!1,V={},B=null,d=null,A=!1,X=null,_=()=>{};const D=!e.ownPoster&&(r!=null&&r.current.poster)?Wo(n,r.current.poster,e.place,r,i):null,$=()=>{if(!He||(r==null?void 0:r.current.globalLook)===!1||(b.mark??"glyph")!=="glyph"&&!e.glyphs||qo.test(b.charset??""))return;const g=He.style?co[He.style]:void 0,w=(g==null?void 0:g.mark)??"glyph",C=w==="glyph"?He.charset||(g&&"charset"in g?g.charset:void 0):void 0;return g||C?{mark:w,charset:C}:void 0},re=()=>({...e.glyphs?Object.fromEntries(Object.keys(e.glyphs()).map(g=>[g,b[g]]).filter(([,g])=>g!==void 0)):{},mark:b.mark??"glyph",charset:b.charset??" .,:-=+*#%@",glyphScale:b.glyphScale??.8,gamma:b.gamma??1,fontWeight:b.fontWeight??500}),ae=g=>{if(!g.charset)return{...re(),mark:g.mark};const w=e.glyphs&&(b.mark??"glyph")!=="glyph"?e.glyphs():{};return{...w,mark:g.mark,charset:g.charset,...$o(g.charset,{...b,...w})}},O=()=>{const g=$(),w=g&&`${g.mark}\0${g.charset??""}`;return w===l?null:(l=w,g?ae(g):re())},W={host:n,doc:document,eager:(r==null?void 0:r.current.lazy)===!1,wanted:!1,live:!1,poster:!!D||!!e.ownPoster,mount(){var ne,q,ee;if(v||a)return;m&&(i=e.canvas=Ur(i)),m=!0,b={...e.options(),...h},l=void 0;const g=O();let w;try{w=so(i,{...b,...g,...t===void 0?{}:{source:t},onError:()=>{a===w&&we()}})}catch{return void we()}a=w,W.live=!0,W.shared=Yr(i);const C=i;if(W.shared)_=()=>{};else{let j=0;const ue=()=>{a===w&&(D==null||D.show(!0),clearTimeout(j),j=setTimeout(()=>{a===w&&i===C&&(W.release(),Ve())},1e3))},oe=()=>{clearTimeout(j),requestAnimationFrame(()=>requestAnimationFrame(()=>{a===w&&(D==null||D.show(!1))}))};(ne=C.addEventListener)==null||ne.call(C,"webglcontextlost",ue),(q=C.addEventListener)==null||q.call(C,"webglcontextrestored",oe),_=()=>{var S,me;clearTimeout(j),(S=C.removeEventListener)==null||S.call(C,"webglcontextlost",ue),(me=C.removeEventListener)==null||me.call(C,"webglcontextrestored",oe)}}R&&w.pause(),w.ready.then(()=>{a===w&&(A=!0,D==null||D.show(!1),B==null||B(),B=d=null)},()=>{}),(ee=e.onMount)==null||ee.call(e,w,i)},release(){var w;if(!a)return;const g=a;a=null,W.live=!1,A=!1,_(),X==null||X.clear(),X=null,V=g.settings,g.destroy(),W.shared?v||(i.width=i.width,m=!1):((C=>{var ne,q;try{const ee=C.getContext("webgl2");ee&&!((ne=ee.isContextLost)!=null&&ne.call(ee))&&((q=ee.getExtension("WEBGL_lose_context"))==null||q.loseContext())}catch{}})(i),v||(i=e.canvas=Ur(i),m=!1)),D==null||D.show(!0),(w=e.onRelease)==null||w.call(e)},prewarm(){if(v||a)return;b={...e.options(),...h};const g=$();lo(i,{...b,...g?ae(g):{}})}};function we(){f||(f=!0,pe(),p||e.onFail())}function pe(){v||(v=!0,W.release(),(function(g){le.delete(g)&&([...le].some(w=>w.host===g.host)||(ce==null||ce.list.forEach(w=>w.unobserve(g.host))),!le.size&&ce&&(ce.list.forEach(w=>w.disconnect()),ce=null),le.size||(ho(null),nt&&typeof cancelIdleCallback=="function"&&cancelIdleCallback(nt),nt=0),Ve())})(W),_n.delete(fe),i.remove(),f?D==null||D.show(!0):D==null||D.remove())}const fe={relook(){a||(q=>{Dn.delete(q),le.has(q)&&Xn()})(W);const g=O();if(!g||!a)return;const w=a,C=A&&!((r==null?void 0:r.current.reducedMotion)??(typeof matchMedia=="function"&&matchMedia("(prefers-reduced-motion: reduce)").matches));X==null||X.clear(),X=null,C&&(w.redraw(),X=(function(q,ee){if(!q.width||!q.height||typeof document>"u")return null;const j=document.createElement("canvas");j.width=q.width,j.height=q.height;const ue=j.getContext("2d");if(!ue||typeof j.animate!="function")return null;try{ue.drawImage(q,0,0)}catch{return null}j.setAttribute("aria-hidden","true"),j.dataset.asciiSweep="",j.setAttribute("style",ee);const oe=document.createElement("span");let S,me;oe.setAttribute("aria-hidden","true"),oe.dataset.asciiSweep="",oe.setAttribute("style",`${ee};right:auto;width:2px;margin-left:-1px;opacity:0;background:var(--ascii-sweep, #f3ba32);box-shadow:0 0 10px 1px color-mix(in srgb, var(--ascii-sweep, #f3ba32) 40%, transparent)`),q.after(j,oe);const be=()=>{S==null||S.cancel(),me==null||me.cancel(),j.remove(),oe.remove()};return{clear:be,reveal(){if(!j.isConnected)return;const Ee={duration:200,delay:40,easing:"cubic-bezier(.23,1,.32,1)",fill:"both"};S=j.animate([{clipPath:"inset(0% 0% 0% 0%)"},{clipPath:"inset(0% 0% 0% 100%)"}],Ee);const ke=q.clientWidth||q.getBoundingClientRect().width;me=oe.animate([{transform:"translateX(0px)",opacity:.35},{opacity:.35,offset:.7},{transform:`translateX(${ke}px)`,opacity:0}],Ee),S.onfinish=be}}})(i,e.place)),w.update(g);const ne=X;ne&&requestAnimationFrame(()=>requestAnimationFrame(()=>{X===ne&&a===w?ne.reveal():ne.clear()}))}};return{field:{get canvas(){return i},get ready(){return a?a.ready:d??(d=new Promise(g=>{B=g}))},get settings(){return(a==null?void 0:a.settings)??V},update(g){if(h={...h,...g},b={...b,...g},!a)return;const w=O(),C=$();a.update(C?{...g,...ae(C)}:w?{...g,...w}:g)},setSource:g=>(t=g,a?a.setSource(g):Promise.resolve()),pointer(g,w){a==null||a.pointer(g,w)},redraw(){a==null||a.redraw()},pulse(g){a==null||a.pulse(g)},play(){R=!1,a==null||a.play()},pause(){R=!0,a==null||a.pause()},resize(){a==null||a.resize()},destroy(){pe()}},get canvas(){return i},get live(){return!!a},get failed(){return f},start(){_n.add(fe),p=!0;try{Vo(W)}finally{p=!1}},destroy:pe}}export{Zo as C,An as M,Io as O,ta as Q,Sn as R,ea as S,Gr as T,na as U,Or as V,zr as W,_o as Y,Wo as Z,Yo as f,Jo as h,Hr as j,Qo as m,co as o,ra as t,Ko as u};
