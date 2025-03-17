/* A Bison parser, made by GNU Bison 3.8.2.  */

/* Bison implementation for Yacc-like parsers in C

   Copyright (C) 1984, 1989-1990, 2000-2015, 2018-2021 Free Software Foundation,
   Inc.

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <https://www.gnu.org/licenses/>.  */

/* As a special exception, you may create a larger work that contains
   part or all of the Bison parser skeleton and distribute that work
   under terms of your choice, so long as that work isn't itself a
   parser generator using the skeleton or a modified version thereof
   as a parser skeleton.  Alternatively, if you modify or redistribute
   the parser skeleton itself, you may (at your option) remove this
   special exception, which will cause the skeleton and the resulting
   Bison output files to be licensed under the GNU General Public
   License without this special exception.

   This special exception was added by the Free Software Foundation in
   version 2.2 of Bison.  */

/* C LALR(1) parser skeleton written by Richard Stallman, by
   simplifying the original so-called "semantic" parser.  */

/* DO NOT RELY ON FEATURES THAT ARE NOT DOCUMENTED in the manual,
   especially those whose name start with YY_ or yy_.  They are
   private implementation details that can be changed or removed.  */

/* All symbols defined below should begin with yy or YY, to avoid
   infringing on user name space.  This should be done even for local
   variables, as they might otherwise be expanded by user macros.
   There are some unavoidable exceptions within include files to
   define necessary library symbols; they are noted "INFRINGES ON
   USER NAME SPACE" below.  */

/* Identify Bison output, and Bison version.  */
#define YYBISON 30802

/* Bison version string.  */
#define YYBISON_VERSION "3.8.2"

/* Skeleton name.  */
#define YYSKELETON_NAME "yacc.c"

/* Pure parsers.  */
#define YYPURE 0

/* Push parsers.  */
#define YYPUSH 0

/* Pull parsers.  */
#define YYPULL 1




/* First part of user prologue.  */
#line 37 "scan.y"

  #include <stdio.h>
  #include <stdlib.h>
  #ifdef MACOS
   #include <malloc/malloc.h>
  #endif
  /* #include <malloc.h> not necessary, "malloc" comes from <stdlib.h> */
  #include <string.h>
  #include <unistd.h>
  #include "scan.h"

  #define __YYSCLASS

  #define YYDEBUG 1
  extern char yytext[];
  extern FILE * yyin;
  
  int nError   = 0;
  int nWarning = 0;

  int crt_section;
  int eqState;
  int isPhoto = 0;

  char crt_term[ 30 ];
  char crt_coef[ 30 ];

  char * InlineBuf;
  int InlineLen;

  void SemicolonError();
  int yyerrflag=0;

  void ParserErrorMessage();
  void yyerror(char *);


#line 109 "y.tab.c"

# ifndef YY_CAST
#  ifdef __cplusplus
#   define YY_CAST(Type, Val) static_cast<Type> (Val)
#   define YY_REINTERPRET_CAST(Type, Val) reinterpret_cast<Type> (Val)
#  else
#   define YY_CAST(Type, Val) ((Type) (Val))
#   define YY_REINTERPRET_CAST(Type, Val) ((Type) (Val))
#  endif
# endif
# ifndef YY_NULLPTR
#  if defined __cplusplus
#   if 201103L <= __cplusplus
#    define YY_NULLPTR nullptr
#   else
#    define YY_NULLPTR 0
#   endif
#  else
#   define YY_NULLPTR ((void*)0)
#  endif
# endif

/* Use api.header.include to #include this header
   instead of duplicating it here.  */
#ifndef YY_YY_Y_TAB_H_INCLUDED
# define YY_YY_Y_TAB_H_INCLUDED
/* Debug traces.  */
#ifndef YYDEBUG
# define YYDEBUG 0
#endif
#if YYDEBUG
extern int yydebug;
#endif

/* Token kinds.  */
#ifndef YYTOKENTYPE
# define YYTOKENTYPE
  enum yytokentype
  {
    YYEMPTY = -2,
    YYEOF = 0,                     /* "end of file"  */
    YYerror = 256,                 /* error  */
    YYUNDEF = 257,                 /* "invalid token"  */
    JACOBIAN = 258,                /* JACOBIAN  */
    DOUBLE = 259,                  /* DOUBLE  */
    FUNCTION = 260,                /* FUNCTION  */
    DEFVAR = 261,                  /* DEFVAR  */
    DEFRAD = 262,                  /* DEFRAD  */
    DEFFIX = 263,                  /* DEFFIX  */
    SETVAR = 264,                  /* SETVAR  */
    SETRAD = 265,                  /* SETRAD  */
    SETFIX = 266,                  /* SETFIX  */
    HESSIAN = 267,                 /* HESSIAN  */
    STOICMAT = 268,                /* STOICMAT  */
    STOCHASTIC = 269,              /* STOCHASTIC  */
    DECLARE = 270,                 /* DECLARE  */
    INITVALUES = 271,              /* INITVALUES  */
    EQUATIONS = 272,               /* EQUATIONS  */
    FAMILIES = 273,                /* FAMILIES  */
    LUMP = 274,                    /* LUMP  */
    INIEQUAL = 275,                /* INIEQUAL  */
    EQNEQUAL = 276,                /* EQNEQUAL  */
    EQNCOLON = 277,                /* EQNCOLON  */
    LMPCOLON = 278,                /* LMPCOLON  */
    LMPPLUS = 279,                 /* LMPPLUS  */
    SPCPLUS = 280,                 /* SPCPLUS  */
    SPCEQUAL = 281,                /* SPCEQUAL  */
    FAMCOLON = 282,                /* FAMCOLON  */
    ATOMDECL = 283,                /* ATOMDECL  */
    CHECK = 284,                   /* CHECK  */
    CHECKALL = 285,                /* CHECKALL  */
    REORDER = 286,                 /* REORDER  */
    MEX = 287,                     /* MEX  */
    DUMMYINDEX = 288,              /* DUMMYINDEX  */
    EQNTAGS = 289,                 /* EQNTAGS  */
    LOOKAT = 290,                  /* LOOKAT  */
    LOOKATALL = 291,               /* LOOKATALL  */
    TRANSPORT = 292,               /* TRANSPORT  */
    TRANSPORTALL = 293,            /* TRANSPORTALL  */
    MONITOR = 294,                 /* MONITOR  */
    USES = 295,                    /* USES  */
    SPARSEDATA = 296,              /* SPARSEDATA  */
    WRITE_ATM = 297,               /* WRITE_ATM  */
    WRITE_SPC = 298,               /* WRITE_SPC  */
    WRITE_MAT = 299,               /* WRITE_MAT  */
    WRITE_OPT = 300,               /* WRITE_OPT  */
    INITIALIZE = 301,              /* INITIALIZE  */
    XGRID = 302,                   /* XGRID  */
    YGRID = 303,                   /* YGRID  */
    ZGRID = 304,                   /* ZGRID  */
    USE = 305,                     /* USE  */
    LANGUAGE = 306,                /* LANGUAGE  */
    INTFILE = 307,                 /* INTFILE  */
    DRIVER = 308,                  /* DRIVER  */
    RUN = 309,                     /* RUN  */
    INLINE = 310,                  /* INLINE  */
    ENDINLINE = 311,               /* ENDINLINE  */
    PARAMETER = 312,               /* PARAMETER  */
    SPCSPC = 313,                  /* SPCSPC  */
    INISPC = 314,                  /* INISPC  */
    INIVALUE = 315,                /* INIVALUE  */
    EQNSPC = 316,                  /* EQNSPC  */
    EQNSIGN = 317,                 /* EQNSIGN  */
    EQNCOEF = 318,                 /* EQNCOEF  */
    RATE = 319,                    /* RATE  */
    LMPSPC = 320,                  /* LMPSPC  */
    SPCNR = 321,                   /* SPCNR  */
    ATOMID = 322,                  /* ATOMID  */
    LKTID = 323,                   /* LKTID  */
    MNIID = 324,                   /* MNIID  */
    INLCTX = 325,                  /* INLCTX  */
    INCODE = 326,                  /* INCODE  */
    SSPID = 327,                   /* SSPID  */
    EQNLESS = 328,                 /* EQNLESS  */
    EQNTAG = 329,                  /* EQNTAG  */
    EQNGREATER = 330,              /* EQNGREATER  */
    TPTID = 331,                   /* TPTID  */
    USEID = 332,                   /* USEID  */
    FLUX = 333,                    /* FLUX  */
    AUTOREDUCE = 334,              /* AUTOREDUCE  */
    PLSPC = 335,                   /* PLSPC  */
    UPPERCASEF90 = 336,            /* UPPERCASEF90  */
    MINVERSION = 337               /* MINVERSION  */
  };
  typedef enum yytokentype yytoken_kind_t;
#endif

/* Value type.  */
#if ! defined YYSTYPE && ! defined YYSTYPE_IS_DECLARED
union YYSTYPE
{
#line 76 "scan.y"

  char str[1000];

#line 245 "y.tab.c"

};
typedef union YYSTYPE YYSTYPE;
# define YYSTYPE_IS_TRIVIAL 1
# define YYSTYPE_IS_DECLARED 1
#endif


extern YYSTYPE yylval;


int yyparse (void);


#endif /* !YY_YY_Y_TAB_H_INCLUDED  */
/* Symbol kind.  */
enum yysymbol_kind_t
{
  YYSYMBOL_YYEMPTY = -2,
  YYSYMBOL_YYEOF = 0,                      /* "end of file"  */
  YYSYMBOL_YYerror = 1,                    /* error  */
  YYSYMBOL_YYUNDEF = 2,                    /* "invalid token"  */
  YYSYMBOL_JACOBIAN = 3,                   /* JACOBIAN  */
  YYSYMBOL_DOUBLE = 4,                     /* DOUBLE  */
  YYSYMBOL_FUNCTION = 5,                   /* FUNCTION  */
  YYSYMBOL_DEFVAR = 6,                     /* DEFVAR  */
  YYSYMBOL_DEFRAD = 7,                     /* DEFRAD  */
  YYSYMBOL_DEFFIX = 8,                     /* DEFFIX  */
  YYSYMBOL_SETVAR = 9,                     /* SETVAR  */
  YYSYMBOL_SETRAD = 10,                    /* SETRAD  */
  YYSYMBOL_SETFIX = 11,                    /* SETFIX  */
  YYSYMBOL_HESSIAN = 12,                   /* HESSIAN  */
  YYSYMBOL_STOICMAT = 13,                  /* STOICMAT  */
  YYSYMBOL_STOCHASTIC = 14,                /* STOCHASTIC  */
  YYSYMBOL_DECLARE = 15,                   /* DECLARE  */
  YYSYMBOL_INITVALUES = 16,                /* INITVALUES  */
  YYSYMBOL_EQUATIONS = 17,                 /* EQUATIONS  */
  YYSYMBOL_FAMILIES = 18,                  /* FAMILIES  */
  YYSYMBOL_LUMP = 19,                      /* LUMP  */
  YYSYMBOL_INIEQUAL = 20,                  /* INIEQUAL  */
  YYSYMBOL_EQNEQUAL = 21,                  /* EQNEQUAL  */
  YYSYMBOL_EQNCOLON = 22,                  /* EQNCOLON  */
  YYSYMBOL_LMPCOLON = 23,                  /* LMPCOLON  */
  YYSYMBOL_LMPPLUS = 24,                   /* LMPPLUS  */
  YYSYMBOL_SPCPLUS = 25,                   /* SPCPLUS  */
  YYSYMBOL_SPCEQUAL = 26,                  /* SPCEQUAL  */
  YYSYMBOL_FAMCOLON = 27,                  /* FAMCOLON  */
  YYSYMBOL_ATOMDECL = 28,                  /* ATOMDECL  */
  YYSYMBOL_CHECK = 29,                     /* CHECK  */
  YYSYMBOL_CHECKALL = 30,                  /* CHECKALL  */
  YYSYMBOL_REORDER = 31,                   /* REORDER  */
  YYSYMBOL_MEX = 32,                       /* MEX  */
  YYSYMBOL_DUMMYINDEX = 33,                /* DUMMYINDEX  */
  YYSYMBOL_EQNTAGS = 34,                   /* EQNTAGS  */
  YYSYMBOL_LOOKAT = 35,                    /* LOOKAT  */
  YYSYMBOL_LOOKATALL = 36,                 /* LOOKATALL  */
  YYSYMBOL_TRANSPORT = 37,                 /* TRANSPORT  */
  YYSYMBOL_TRANSPORTALL = 38,              /* TRANSPORTALL  */
  YYSYMBOL_MONITOR = 39,                   /* MONITOR  */
  YYSYMBOL_USES = 40,                      /* USES  */
  YYSYMBOL_SPARSEDATA = 41,                /* SPARSEDATA  */
  YYSYMBOL_WRITE_ATM = 42,                 /* WRITE_ATM  */
  YYSYMBOL_WRITE_SPC = 43,                 /* WRITE_SPC  */
  YYSYMBOL_WRITE_MAT = 44,                 /* WRITE_MAT  */
  YYSYMBOL_WRITE_OPT = 45,                 /* WRITE_OPT  */
  YYSYMBOL_INITIALIZE = 46,                /* INITIALIZE  */
  YYSYMBOL_XGRID = 47,                     /* XGRID  */
  YYSYMBOL_YGRID = 48,                     /* YGRID  */
  YYSYMBOL_ZGRID = 49,                     /* ZGRID  */
  YYSYMBOL_USE = 50,                       /* USE  */
  YYSYMBOL_LANGUAGE = 51,                  /* LANGUAGE  */
  YYSYMBOL_INTFILE = 52,                   /* INTFILE  */
  YYSYMBOL_DRIVER = 53,                    /* DRIVER  */
  YYSYMBOL_RUN = 54,                       /* RUN  */
  YYSYMBOL_INLINE = 55,                    /* INLINE  */
  YYSYMBOL_ENDINLINE = 56,                 /* ENDINLINE  */
  YYSYMBOL_PARAMETER = 57,                 /* PARAMETER  */
  YYSYMBOL_SPCSPC = 58,                    /* SPCSPC  */
  YYSYMBOL_INISPC = 59,                    /* INISPC  */
  YYSYMBOL_INIVALUE = 60,                  /* INIVALUE  */
  YYSYMBOL_EQNSPC = 61,                    /* EQNSPC  */
  YYSYMBOL_EQNSIGN = 62,                   /* EQNSIGN  */
  YYSYMBOL_EQNCOEF = 63,                   /* EQNCOEF  */
  YYSYMBOL_RATE = 64,                      /* RATE  */
  YYSYMBOL_LMPSPC = 65,                    /* LMPSPC  */
  YYSYMBOL_SPCNR = 66,                     /* SPCNR  */
  YYSYMBOL_ATOMID = 67,                    /* ATOMID  */
  YYSYMBOL_LKTID = 68,                     /* LKTID  */
  YYSYMBOL_MNIID = 69,                     /* MNIID  */
  YYSYMBOL_INLCTX = 70,                    /* INLCTX  */
  YYSYMBOL_INCODE = 71,                    /* INCODE  */
  YYSYMBOL_SSPID = 72,                     /* SSPID  */
  YYSYMBOL_EQNLESS = 73,                   /* EQNLESS  */
  YYSYMBOL_EQNTAG = 74,                    /* EQNTAG  */
  YYSYMBOL_EQNGREATER = 75,                /* EQNGREATER  */
  YYSYMBOL_TPTID = 76,                     /* TPTID  */
  YYSYMBOL_USEID = 77,                     /* USEID  */
  YYSYMBOL_FLUX = 78,                      /* FLUX  */
  YYSYMBOL_AUTOREDUCE = 79,                /* AUTOREDUCE  */
  YYSYMBOL_PLSPC = 80,                     /* PLSPC  */
  YYSYMBOL_UPPERCASEF90 = 81,              /* UPPERCASEF90  */
  YYSYMBOL_MINVERSION = 82,                /* MINVERSION  */
  YYSYMBOL_83_ = 83,                       /* ';'  */
  YYSYMBOL_YYACCEPT = 84,                  /* $accept  */
  YYSYMBOL_program = 85,                   /* program  */
  YYSYMBOL_section = 86,                   /* section  */
  YYSYMBOL_semicolon = 87,                 /* semicolon  */
  YYSYMBOL_atomlist = 88,                  /* atomlist  */
  YYSYMBOL_atomdef = 89,                   /* atomdef  */
  YYSYMBOL_lookatlist = 90,                /* lookatlist  */
  YYSYMBOL_lookatspc = 91,                 /* lookatspc  */
  YYSYMBOL_monitorlist = 92,               /* monitorlist  */
  YYSYMBOL_monitorspc = 93,                /* monitorspc  */
  YYSYMBOL_translist = 94,                 /* translist  */
  YYSYMBOL_transspc = 95,                  /* transspc  */
  YYSYMBOL_uselist = 96,                   /* uselist  */
  YYSYMBOL_usefile = 97,                   /* usefile  */
  YYSYMBOL_setspclist = 98,                /* setspclist  */
  YYSYMBOL_setspcspc = 99,                 /* setspcspc  */
  YYSYMBOL_families = 100,                 /* families  */
  YYSYMBOL_family = 101,                   /* family  */
  YYSYMBOL_members = 102,                  /* members  */
  YYSYMBOL_member = 103,                   /* member  */
  YYSYMBOL_species = 104,                  /* species  */
  YYSYMBOL_spc = 105,                      /* spc  */
  YYSYMBOL_spcname = 106,                  /* spcname  */
  YYSYMBOL_spcdef = 107,                   /* spcdef  */
  YYSYMBOL_atoms = 108,                    /* atoms  */
  YYSYMBOL_atom = 109,                     /* atom  */
  YYSYMBOL_initvalues = 110,               /* initvalues  */
  YYSYMBOL_assignment = 111,               /* assignment  */
  YYSYMBOL_equations = 112,                /* equations  */
  YYSYMBOL_equation = 113,                 /* equation  */
  YYSYMBOL_rate = 114,                     /* rate  */
  YYSYMBOL_eqntag = 115,                   /* eqntag  */
  YYSYMBOL_lefths = 116,                   /* lefths  */
  YYSYMBOL_righths = 117,                  /* righths  */
  YYSYMBOL_expresion = 118,                /* expresion  */
  YYSYMBOL_term = 119,                     /* term  */
  YYSYMBOL_lumps = 120,                    /* lumps  */
  YYSYMBOL_lump = 121,                     /* lump  */
  YYSYMBOL_inlinecode = 122                /* inlinecode  */
};
typedef enum yysymbol_kind_t yysymbol_kind_t;




#ifdef short
# undef short
#endif

/* On compilers that do not define __PTRDIFF_MAX__ etc., make sure
   <limits.h> and (if available) <stdint.h> are included
   so that the code can choose integer types of a good width.  */

#ifndef __PTRDIFF_MAX__
# include <limits.h> /* INFRINGES ON USER NAME SPACE */
# if defined __STDC_VERSION__ && 199901 <= __STDC_VERSION__
#  include <stdint.h> /* INFRINGES ON USER NAME SPACE */
#  define YY_STDINT_H
# endif
#endif

/* Narrow types that promote to a signed type and that can represent a
   signed or unsigned integer of at least N bits.  In tables they can
   save space and decrease cache pressure.  Promoting to a signed type
   helps avoid bugs in integer arithmetic.  */

#ifdef __INT_LEAST8_MAX__
typedef __INT_LEAST8_TYPE__ yytype_int8;
#elif defined YY_STDINT_H
typedef int_least8_t yytype_int8;
#else
typedef signed char yytype_int8;
#endif

#ifdef __INT_LEAST16_MAX__
typedef __INT_LEAST16_TYPE__ yytype_int16;
#elif defined YY_STDINT_H
typedef int_least16_t yytype_int16;
#else
typedef short yytype_int16;
#endif

/* Work around bug in HP-UX 11.23, which defines these macros
   incorrectly for preprocessor constants.  This workaround can likely
   be removed in 2023, as HPE has promised support for HP-UX 11.23
   (aka HP-UX 11i v2) only through the end of 2022; see Table 2 of
   <https://h20195.www2.hpe.com/V2/getpdf.aspx/4AA4-7673ENW.pdf>.  */
#ifdef __hpux
# undef UINT_LEAST8_MAX
# undef UINT_LEAST16_MAX
# define UINT_LEAST8_MAX 255
# define UINT_LEAST16_MAX 65535
#endif

#if defined __UINT_LEAST8_MAX__ && __UINT_LEAST8_MAX__ <= __INT_MAX__
typedef __UINT_LEAST8_TYPE__ yytype_uint8;
#elif (!defined __UINT_LEAST8_MAX__ && defined YY_STDINT_H \
       && UINT_LEAST8_MAX <= INT_MAX)
typedef uint_least8_t yytype_uint8;
#elif !defined __UINT_LEAST8_MAX__ && UCHAR_MAX <= INT_MAX
typedef unsigned char yytype_uint8;
#else
typedef short yytype_uint8;
#endif

#if defined __UINT_LEAST16_MAX__ && __UINT_LEAST16_MAX__ <= __INT_MAX__
typedef __UINT_LEAST16_TYPE__ yytype_uint16;
#elif (!defined __UINT_LEAST16_MAX__ && defined YY_STDINT_H \
       && UINT_LEAST16_MAX <= INT_MAX)
typedef uint_least16_t yytype_uint16;
#elif !defined __UINT_LEAST16_MAX__ && USHRT_MAX <= INT_MAX
typedef unsigned short yytype_uint16;
#else
typedef int yytype_uint16;
#endif

#ifndef YYPTRDIFF_T
# if defined __PTRDIFF_TYPE__ && defined __PTRDIFF_MAX__
#  define YYPTRDIFF_T __PTRDIFF_TYPE__
#  define YYPTRDIFF_MAXIMUM __PTRDIFF_MAX__
# elif defined PTRDIFF_MAX
#  ifndef ptrdiff_t
#   include <stddef.h> /* INFRINGES ON USER NAME SPACE */
#  endif
#  define YYPTRDIFF_T ptrdiff_t
#  define YYPTRDIFF_MAXIMUM PTRDIFF_MAX
# else
#  define YYPTRDIFF_T long
#  define YYPTRDIFF_MAXIMUM LONG_MAX
# endif
#endif

#ifndef YYSIZE_T
# ifdef __SIZE_TYPE__
#  define YYSIZE_T __SIZE_TYPE__
# elif defined size_t
#  define YYSIZE_T size_t
# elif defined __STDC_VERSION__ && 199901 <= __STDC_VERSION__
#  include <stddef.h> /* INFRINGES ON USER NAME SPACE */
#  define YYSIZE_T size_t
# else
#  define YYSIZE_T unsigned
# endif
#endif

#define YYSIZE_MAXIMUM                                  \
  YY_CAST (YYPTRDIFF_T,                                 \
           (YYPTRDIFF_MAXIMUM < YY_CAST (YYSIZE_T, -1)  \
            ? YYPTRDIFF_MAXIMUM                         \
            : YY_CAST (YYSIZE_T, -1)))

#define YYSIZEOF(X) YY_CAST (YYPTRDIFF_T, sizeof (X))


/* Stored state numbers (used for stacks). */
typedef yytype_uint8 yy_state_t;

/* State numbers in computations.  */
typedef int yy_state_fast_t;

#ifndef YY_
# if defined YYENABLE_NLS && YYENABLE_NLS
#  if ENABLE_NLS
#   include <libintl.h> /* INFRINGES ON USER NAME SPACE */
#   define YY_(Msgid) dgettext ("bison-runtime", Msgid)
#  endif
# endif
# ifndef YY_
#  define YY_(Msgid) Msgid
# endif
#endif


#ifndef YY_ATTRIBUTE_PURE
# if defined __GNUC__ && 2 < __GNUC__ + (96 <= __GNUC_MINOR__)
#  define YY_ATTRIBUTE_PURE __attribute__ ((__pure__))
# else
#  define YY_ATTRIBUTE_PURE
# endif
#endif

#ifndef YY_ATTRIBUTE_UNUSED
# if defined __GNUC__ && 2 < __GNUC__ + (7 <= __GNUC_MINOR__)
#  define YY_ATTRIBUTE_UNUSED __attribute__ ((__unused__))
# else
#  define YY_ATTRIBUTE_UNUSED
# endif
#endif

/* Suppress unused-variable warnings by "using" E.  */
#if ! defined lint || defined __GNUC__
# define YY_USE(E) ((void) (E))
#else
# define YY_USE(E) /* empty */
#endif

/* Suppress an incorrect diagnostic about yylval being uninitialized.  */
#if defined __GNUC__ && ! defined __ICC && 406 <= __GNUC__ * 100 + __GNUC_MINOR__
# if __GNUC__ * 100 + __GNUC_MINOR__ < 407
#  define YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN                           \
    _Pragma ("GCC diagnostic push")                                     \
    _Pragma ("GCC diagnostic ignored \"-Wuninitialized\"")
# else
#  define YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN                           \
    _Pragma ("GCC diagnostic push")                                     \
    _Pragma ("GCC diagnostic ignored \"-Wuninitialized\"")              \
    _Pragma ("GCC diagnostic ignored \"-Wmaybe-uninitialized\"")
# endif
# define YY_IGNORE_MAYBE_UNINITIALIZED_END      \
    _Pragma ("GCC diagnostic pop")
#else
# define YY_INITIAL_VALUE(Value) Value
#endif
#ifndef YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
# define YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
# define YY_IGNORE_MAYBE_UNINITIALIZED_END
#endif
#ifndef YY_INITIAL_VALUE
# define YY_INITIAL_VALUE(Value) /* Nothing. */
#endif

#if defined __cplusplus && defined __GNUC__ && ! defined __ICC && 6 <= __GNUC__
# define YY_IGNORE_USELESS_CAST_BEGIN                          \
    _Pragma ("GCC diagnostic push")                            \
    _Pragma ("GCC diagnostic ignored \"-Wuseless-cast\"")
# define YY_IGNORE_USELESS_CAST_END            \
    _Pragma ("GCC diagnostic pop")
#endif
#ifndef YY_IGNORE_USELESS_CAST_BEGIN
# define YY_IGNORE_USELESS_CAST_BEGIN
# define YY_IGNORE_USELESS_CAST_END
#endif


#define YY_ASSERT(E) ((void) (0 && (E)))

#if !defined yyoverflow

/* The parser invokes alloca or malloc; define the necessary symbols.  */

# ifdef YYSTACK_USE_ALLOCA
#  if YYSTACK_USE_ALLOCA
#   ifdef __GNUC__
#    define YYSTACK_ALLOC __builtin_alloca
#   elif defined __BUILTIN_VA_ARG_INCR
#    include <alloca.h> /* INFRINGES ON USER NAME SPACE */
#   elif defined _AIX
#    define YYSTACK_ALLOC __alloca
#   elif defined _MSC_VER
#    include <malloc.h> /* INFRINGES ON USER NAME SPACE */
#    define alloca _alloca
#   else
#    define YYSTACK_ALLOC alloca
#    if ! defined _ALLOCA_H && ! defined EXIT_SUCCESS
#     include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
      /* Use EXIT_SUCCESS as a witness for stdlib.h.  */
#     ifndef EXIT_SUCCESS
#      define EXIT_SUCCESS 0
#     endif
#    endif
#   endif
#  endif
# endif

# ifdef YYSTACK_ALLOC
   /* Pacify GCC's 'empty if-body' warning.  */
#  define YYSTACK_FREE(Ptr) do { /* empty */; } while (0)
#  ifndef YYSTACK_ALLOC_MAXIMUM
    /* The OS might guarantee only one guard page at the bottom of the stack,
       and a page size can be as small as 4096 bytes.  So we cannot safely
       invoke alloca (N) if N exceeds 4096.  Use a slightly smaller number
       to allow for a few compiler-allocated temporary stack slots.  */
#   define YYSTACK_ALLOC_MAXIMUM 4032 /* reasonable circa 2006 */
#  endif
# else
#  define YYSTACK_ALLOC YYMALLOC
#  define YYSTACK_FREE YYFREE
#  ifndef YYSTACK_ALLOC_MAXIMUM
#   define YYSTACK_ALLOC_MAXIMUM YYSIZE_MAXIMUM
#  endif
#  if (defined __cplusplus && ! defined EXIT_SUCCESS \
       && ! ((defined YYMALLOC || defined malloc) \
             && (defined YYFREE || defined free)))
#   include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
#   ifndef EXIT_SUCCESS
#    define EXIT_SUCCESS 0
#   endif
#  endif
#  ifndef YYMALLOC
#   define YYMALLOC malloc
#   if ! defined malloc && ! defined EXIT_SUCCESS
void *malloc (YYSIZE_T); /* INFRINGES ON USER NAME SPACE */
#   endif
#  endif
#  ifndef YYFREE
#   define YYFREE free
#   if ! defined free && ! defined EXIT_SUCCESS
void free (void *); /* INFRINGES ON USER NAME SPACE */
#   endif
#  endif
# endif
#endif /* !defined yyoverflow */

#if (! defined yyoverflow \
     && (! defined __cplusplus \
         || (defined YYSTYPE_IS_TRIVIAL && YYSTYPE_IS_TRIVIAL)))

/* A type that is properly aligned for any stack member.  */
union yyalloc
{
  yy_state_t yyss_alloc;
  YYSTYPE yyvs_alloc;
};

/* The size of the maximum gap between one aligned stack and the next.  */
# define YYSTACK_GAP_MAXIMUM (YYSIZEOF (union yyalloc) - 1)

/* The size of an array large to enough to hold all stacks, each with
   N elements.  */
# define YYSTACK_BYTES(N) \
     ((N) * (YYSIZEOF (yy_state_t) + YYSIZEOF (YYSTYPE)) \
      + YYSTACK_GAP_MAXIMUM)

# define YYCOPY_NEEDED 1

/* Relocate STACK from its old location to the new one.  The
   local variables YYSIZE and YYSTACKSIZE give the old and new number of
   elements in the stack, and YYPTR gives the new location of the
   stack.  Advance YYPTR to a properly aligned location for the next
   stack.  */
# define YYSTACK_RELOCATE(Stack_alloc, Stack)                           \
    do                                                                  \
      {                                                                 \
        YYPTRDIFF_T yynewbytes;                                         \
        YYCOPY (&yyptr->Stack_alloc, Stack, yysize);                    \
        Stack = &yyptr->Stack_alloc;                                    \
        yynewbytes = yystacksize * YYSIZEOF (*Stack) + YYSTACK_GAP_MAXIMUM; \
        yyptr += yynewbytes / YYSIZEOF (*yyptr);                        \
      }                                                                 \
    while (0)

#endif

#if defined YYCOPY_NEEDED && YYCOPY_NEEDED
/* Copy COUNT objects from SRC to DST.  The source and destination do
   not overlap.  */
# ifndef YYCOPY
#  if defined __GNUC__ && 1 < __GNUC__
#   define YYCOPY(Dst, Src, Count) \
      __builtin_memcpy (Dst, Src, YY_CAST (YYSIZE_T, (Count)) * sizeof (*(Src)))
#  else
#   define YYCOPY(Dst, Src, Count)              \
      do                                        \
        {                                       \
          YYPTRDIFF_T yyi;                      \
          for (yyi = 0; yyi < (Count); yyi++)   \
            (Dst)[yyi] = (Src)[yyi];            \
        }                                       \
      while (0)
#  endif
# endif
#endif /* !YYCOPY_NEEDED */

/* YYFINAL -- State number of the termination state.  */
#define YYFINAL  137
/* YYLAST -- Last index in YYTABLE.  */
#define YYLAST   215

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  84
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  39
/* YYNRULES -- Number of rules.  */
#define YYNRULES  124
/* YYNSTATES -- Number of states.  */
#define YYNSTATES  227

/* YYMAXUTOK -- Last valid token kind.  */
#define YYMAXUTOK   337


/* YYTRANSLATE(TOKEN-NUM) -- Symbol number corresponding to TOKEN-NUM
   as returned by yylex, with out-of-bounds checking.  */
#define YYTRANSLATE(YYX)                                \
  (0 <= (YYX) && (YYX) <= YYMAXUTOK                     \
   ? YY_CAST (yysymbol_kind_t, yytranslate[YYX])        \
   : YYSYMBOL_YYUNDEF)

/* YYTRANSLATE[TOKEN-NUM] -- Symbol number corresponding to TOKEN-NUM
   as returned by yylex.  */
static const yytype_int8 yytranslate[] =
{
       0,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,    83,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     1,     2,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12,    13,    14,
      15,    16,    17,    18,    19,    20,    21,    22,    23,    24,
      25,    26,    27,    28,    29,    30,    31,    32,    33,    34,
      35,    36,    37,    38,    39,    40,    41,    42,    43,    44,
      45,    46,    47,    48,    49,    50,    51,    52,    53,    54,
      55,    56,    57,    58,    59,    60,    61,    62,    63,    64,
      65,    66,    67,    68,    69,    70,    71,    72,    73,    74,
      75,    76,    77,    78,    79,    80,    81,    82
};

#if YYDEBUG
/* YYRLINE[YYN] -- Source line where rule number YYN was defined.  */
static const yytype_int16 yyrline[] =
{
       0,   104,   104,   105,   107,   110,   113,   116,   119,   122,
     125,   128,   131,   134,   137,   140,   142,   144,   146,   148,
     150,   152,   154,   156,   158,   160,   162,   164,   166,   168,
     170,   172,   174,   176,   178,   180,   182,   184,   186,   188,
     190,   192,   194,   196,   201,   203,   205,   207,   209,   211,
     213,   215,   217,   219,   222,   225,   227,   228,   229,   232,
     239,   240,   241,   244,   248,   249,   250,   253,   257,   258,
     259,   262,   266,   267,   268,   271,   275,   276,   277,   280,
     288,   289,   290,   293,   300,   301,   303,   306,   311,   312,
     313,   316,   317,   319,   327,   335,   336,   338,   341,   345,
     346,   347,   350,   353,   354,   355,   360,   365,   370,   374,
     378,   382,   385,   389,   392,   395,   399,   403,   408,   409,
     410,   413,   416,   421,   425
};
#endif

/** Accessing symbol of state STATE.  */
#define YY_ACCESSING_SYMBOL(State) YY_CAST (yysymbol_kind_t, yystos[State])

#if YYDEBUG || 0
/* The user-facing name of the symbol whose (internal) number is
   YYSYMBOL.  No bounds checking.  */
static const char *yysymbol_name (yysymbol_kind_t yysymbol) YY_ATTRIBUTE_UNUSED;

/* YYTNAME[SYMBOL-NUM] -- String name of the symbol SYMBOL-NUM.
   First, the terminals, then, starting at YYNTOKENS, nonterminals.  */
static const char *const yytname[] =
{
  "\"end of file\"", "error", "\"invalid token\"", "JACOBIAN", "DOUBLE",
  "FUNCTION", "DEFVAR", "DEFRAD", "DEFFIX", "SETVAR", "SETRAD", "SETFIX",
  "HESSIAN", "STOICMAT", "STOCHASTIC", "DECLARE", "INITVALUES",
  "EQUATIONS", "FAMILIES", "LUMP", "INIEQUAL", "EQNEQUAL", "EQNCOLON",
  "LMPCOLON", "LMPPLUS", "SPCPLUS", "SPCEQUAL", "FAMCOLON", "ATOMDECL",
  "CHECK", "CHECKALL", "REORDER", "MEX", "DUMMYINDEX", "EQNTAGS", "LOOKAT",
  "LOOKATALL", "TRANSPORT", "TRANSPORTALL", "MONITOR", "USES",
  "SPARSEDATA", "WRITE_ATM", "WRITE_SPC", "WRITE_MAT", "WRITE_OPT",
  "INITIALIZE", "XGRID", "YGRID", "ZGRID", "USE", "LANGUAGE", "INTFILE",
  "DRIVER", "RUN", "INLINE", "ENDINLINE", "PARAMETER", "SPCSPC", "INISPC",
  "INIVALUE", "EQNSPC", "EQNSIGN", "EQNCOEF", "RATE", "LMPSPC", "SPCNR",
  "ATOMID", "LKTID", "MNIID", "INLCTX", "INCODE", "SSPID", "EQNLESS",
  "EQNTAG", "EQNGREATER", "TPTID", "USEID", "FLUX", "AUTOREDUCE", "PLSPC",
  "UPPERCASEF90", "MINVERSION", "';'", "$accept", "program", "section",
  "semicolon", "atomlist", "atomdef", "lookatlist", "lookatspc",
  "monitorlist", "monitorspc", "translist", "transspc", "uselist",
  "usefile", "setspclist", "setspcspc", "families", "family", "members",
  "member", "species", "spc", "spcname", "spcdef", "atoms", "atom",
  "initvalues", "assignment", "equations", "equation", "rate", "eqntag",
  "lefths", "righths", "expresion", "term", "lumps", "lump", "inlinecode", YY_NULLPTR
};

static const char *
yysymbol_name (yysymbol_kind_t yysymbol)
{
  return yytname[yysymbol];
}
#endif

#define YYPACT_NINF (-102)

#define yypact_value_is_default(Yyn) \
  ((Yyn) == YYPACT_NINF)

#define YYTABLE_NINF (-1)

#define yytable_value_is_error(Yyn) \
  0

/* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
   STATE-NUM.  */
static const yytype_int16 yypact[] =
{
     124,   -31,   -12,    33,    10,    10,    10,    13,    13,    13,
      39,    43,    44,    45,    30,     8,    15,    23,    19,    19,
    -102,    46,    47,    48,    50,     6,  -102,     3,  -102,    11,
       1,    51,  -102,  -102,  -102,  -102,    52,    53,    55,    56,
      58,    59,    61,    62,    64,     5,    65,    67,    68,    87,
      60,   124,  -102,  -102,  -102,   -11,   119,    18,   -11,  -102,
    -102,    18,    18,   -11,  -102,    74,   -11,    74,    74,  -102,
    -102,  -102,  -102,   -11,   127,    90,   -11,   -11,  -102,   -21,
     120,    76,   -34,   -11,     4,     4,     2,  -102,   -11,   121,
      93,   -11,   -11,    -2,   115,   -11,   -11,  -102,   116,   -11,
     116,  -102,  -102,  -102,  -102,   -11,  -102,   114,   -11,   -11,
    -102,   108,   -11,   -11,  -102,   117,   -11,   -11,  -102,   110,
     -11,  -102,  -102,  -102,  -102,  -102,  -102,  -102,  -102,  -102,
    -102,  -102,   118,  -102,  -102,  -102,  -102,  -102,  -102,  -102,
     102,   -23,   -11,   102,   102,   -11,   102,   102,   128,   -11,
     102,   102,  -102,  -102,   122,   -11,   102,     4,   126,    -9,
    -102,   -21,   102,   -22,   -11,   102,   102,   129,   115,   -11,
     102,   102,   -11,   102,   102,   -11,   102,   102,   -11,   102,
     102,   -11,   102,   102,   -11,   102,  -102,   -39,  -102,  -102,
     133,   167,  -102,   102,   102,  -102,   102,  -102,   102,   126,
     126,  -102,  -102,  -102,  -102,   135,   170,  -102,   102,  -102,
    -102,   102,   102,   102,   102,   102,   102,  -102,  -102,  -102,
     -23,  -102,  -102,  -102,   -22,  -102,  -102
};

/* YYDEFACT[STATE-NUM] -- Default reduction number in state STATE-NUM.
   Performed when YYTABLE does not specify something else to do.  Zero
   means the default is an error.  */
static const yytype_int8 yydefact[] =
{
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
      30,     0,     0,     0,     0,     0,    31,     0,    32,     0,
       0,     0,    33,    34,    35,    36,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     2,     4,     8,    13,     0,    94,    17,     0,    91,
      92,    18,    19,     0,    79,    20,     0,    21,    22,     5,
       7,    14,     6,     0,     0,    23,     0,     0,   117,     0,
       0,     0,    24,     0,     0,     0,     0,   115,     0,     0,
      25,     0,     0,     0,    26,     0,     0,    59,    15,     0,
      16,     9,    10,    11,    12,     0,    63,    27,     0,     0,
      71,    29,     0,     0,    67,    28,     0,     0,    75,    48,
       0,    49,    39,    40,    41,    42,    37,    38,    45,    46,
      47,    44,     0,    50,    51,    52,    53,     1,     3,    55,
      90,     0,     0,    89,    78,     0,    77,   101,     0,     0,
     100,   105,   114,   116,     0,     0,   104,     0,     0,     0,
     111,     0,    82,     0,     0,    81,   120,     0,     0,     0,
     119,    58,     0,    57,    62,     0,    61,    70,     0,    69,
      66,     0,    65,    74,     0,    73,   124,     0,    54,    98,
       0,    93,    96,    88,    76,   102,    99,   110,   103,     0,
     109,   107,   112,   113,    87,     0,    83,    85,    80,   122,
     121,   118,    56,    60,    68,    64,    72,    43,   123,    97,
       0,   106,   108,    86,     0,    95,    84
};

/* YYPGOTO[NTERM-NUM].  */
static const yytype_int16 yypgoto[] =
{
    -102,   145,  -102,   -58,   179,   -37,  -102,    92,  -102,    85,
    -102,    96,  -102,    82,    84,   -19,  -102,   123,  -102,   -20,
      89,    -5,  -102,  -102,  -102,   -10,  -102,   134,  -102,   130,
    -101,  -102,   131,    54,   -75,   -78,  -102,   -91,  -102
};

/* YYDEFGOTO[NTERM-NUM].  */
static const yytype_uint8 yydefgoto[] =
{
       0,    50,    51,   140,    98,    99,   107,   108,   115,   116,
     111,   112,   119,   120,    65,    66,    90,    91,   206,   207,
      57,    58,    59,    60,   191,   192,    75,    76,    82,    83,
     201,    84,    85,   158,    86,    87,    94,    95,   187
};

/* YYTABLE[YYPACT[STATE-NUM]] -- What to do in state STATE-NUM.  If
   positive, shift that token.  If negative, reduce the rule whose
   number is the opposite.  If YYTABLE_NINF, syntax error.  */
static const yytype_uint8 yytable[] =
{
     143,   152,   117,   169,   109,   144,   131,   105,   146,    77,
     159,    55,   113,   202,    63,   147,    88,   217,   150,   151,
      96,   167,   168,   160,    92,   156,    52,    78,    79,    80,
     162,    73,   218,   165,   166,   189,   204,   170,   171,    81,
      78,   173,    80,   190,   205,    53,   145,   174,   145,   145,
     176,   177,   142,   161,   179,   180,   142,   142,   182,   183,
     137,   172,   185,   172,   161,    78,    79,    80,    56,    78,
      79,    80,   139,    89,   106,   132,    56,   210,   118,   110,
     114,    81,   159,   203,   193,    64,    97,   194,    93,    74,
      54,   196,    67,    68,    61,    62,    69,   198,   221,   222,
      70,    71,    72,   101,   102,   103,   208,   104,   121,   122,
     123,   211,   124,   125,   212,   126,   127,   213,   128,   129,
     214,   130,   133,   215,   134,   135,   216,     1,     2,     3,
       4,     5,     6,     7,     8,     9,    10,    11,    12,    13,
      14,    15,    16,    17,   136,   141,    64,   148,   163,    74,
     154,    89,    18,    19,    20,    21,    22,    23,    24,    25,
      26,    27,    28,    29,    30,    31,    32,    33,    34,    35,
      36,    37,    38,    39,    40,    41,    42,    43,    44,    45,
      93,   153,   106,    97,   110,   188,   114,   118,   195,   186,
     200,   219,   220,   223,   209,   224,   138,   197,   100,   175,
     181,   184,    46,    47,   226,    48,    49,   178,     0,   149,
     225,   199,   155,   164,     0,   157
};

static const yytype_int16 yycheck[] =
{
      58,    79,     1,    94,     1,    63,     1,     1,    66,     1,
      85,     1,     1,    22,     1,    73,     1,    56,    76,    77,
       1,    23,    24,    21,     1,    83,    57,    61,    62,    63,
      88,     1,    71,    91,    92,    58,    58,    95,    96,    73,
      61,    99,    63,    66,    66,    57,    65,   105,    67,    68,
     108,   109,    57,    62,   112,   113,    61,    62,   116,   117,
       0,    98,   120,   100,    62,    61,    62,    63,    58,    61,
      62,    63,    83,    58,    68,    70,    58,   168,    77,    76,
      69,    73,   157,   161,   142,    72,    67,   145,    65,    59,
      57,   149,     8,     9,     5,     6,    57,   155,   199,   200,
      57,    57,    57,    57,    57,    57,   164,    57,    57,    57,
      57,   169,    57,    57,   172,    57,    57,   175,    57,    57,
     178,    57,    57,   181,    57,    57,   184,     3,     4,     5,
       6,     7,     8,     9,    10,    11,    12,    13,    14,    15,
      16,    17,    18,    19,    57,    26,    72,    20,    27,    59,
      74,    58,    28,    29,    30,    31,    32,    33,    34,    35,
      36,    37,    38,    39,    40,    41,    42,    43,    44,    45,
      46,    47,    48,    49,    50,    51,    52,    53,    54,    55,
      65,    61,    68,    67,    76,    83,    69,    77,    60,    71,
      64,    58,    25,    58,    65,    25,    51,    75,    19,   107,
     115,   119,    78,    79,   224,    81,    82,   111,    -1,    75,
     220,   157,    82,    90,    -1,    84
};

/* YYSTOS[STATE-NUM] -- The symbol kind of the accessing symbol of
   state STATE-NUM.  */
static const yytype_int8 yystos[] =
{
       0,     3,     4,     5,     6,     7,     8,     9,    10,    11,
      12,    13,    14,    15,    16,    17,    18,    19,    28,    29,
      30,    31,    32,    33,    34,    35,    36,    37,    38,    39,
      40,    41,    42,    43,    44,    45,    46,    47,    48,    49,
      50,    51,    52,    53,    54,    55,    78,    79,    81,    82,
      85,    86,    57,    57,    57,     1,    58,   104,   105,   106,
     107,   104,   104,     1,    72,    98,    99,    98,    98,    57,
      57,    57,    57,     1,    59,   110,   111,     1,    61,    62,
      63,    73,   112,   113,   115,   116,   118,   119,     1,    58,
     100,   101,     1,    65,   120,   121,     1,    67,    88,    89,
      88,    57,    57,    57,    57,     1,    68,    90,    91,     1,
      76,    94,    95,     1,    69,    92,    93,     1,    77,    96,
      97,    57,    57,    57,    57,    57,    57,    57,    57,    57,
      57,     1,    70,    57,    57,    57,    57,     0,    85,    83,
      87,    26,   105,    87,    87,    99,    87,    87,    20,   111,
      87,    87,   119,    61,    74,   113,    87,   116,   117,   118,
      21,    62,    87,    27,   101,    87,    87,    23,    24,   121,
      87,    87,    89,    87,    87,    91,    87,    87,    95,    87,
      87,    93,    87,    87,    97,    87,    71,   122,    83,    58,
      66,   108,   109,    87,    87,    60,    87,    75,    87,   117,
      64,   114,    22,   119,    58,    66,   102,   103,    87,    65,
     121,    87,    87,    87,    87,    87,    87,    56,    71,    58,
      25,   114,   114,    58,    25,   109,   103
};

/* YYR1[RULE-NUM] -- Symbol kind of the left-hand side of rule RULE-NUM.  */
static const yytype_int8 yyr1[] =
{
       0,    84,    85,    85,    86,    86,    86,    86,    86,    86,
      86,    86,    86,    86,    86,    86,    86,    86,    86,    86,
      86,    86,    86,    86,    86,    86,    86,    86,    86,    86,
      86,    86,    86,    86,    86,    86,    86,    86,    86,    86,
      86,    86,    86,    86,    86,    86,    86,    86,    86,    86,
      86,    86,    86,    86,    87,    87,    88,    88,    88,    89,
      90,    90,    90,    91,    92,    92,    92,    93,    94,    94,
      94,    95,    96,    96,    96,    97,    98,    98,    98,    99,
     100,   100,   100,   101,   102,   102,   103,   103,   104,   104,
     104,   105,   105,   106,   107,   108,   108,   109,   109,   110,
     110,   110,   111,   112,   112,   112,   113,   113,   114,   114,
     115,   116,   117,   118,   118,   118,   119,   119,   120,   120,
     120,   121,   121,   122,   122
};

/* YYR2[RULE-NUM] -- Number of symbols on the right-hand side of rule RULE-NUM.  */
static const yytype_int8 yyr2[] =
{
       0,     2,     1,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       1,     1,     1,     1,     1,     1,     1,     2,     2,     2,
       2,     2,     2,     4,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     1,     3,     2,     2,     1,
       3,     2,     2,     1,     3,     2,     2,     1,     3,     2,
       2,     1,     3,     2,     2,     1,     3,     2,     2,     1,
       3,     2,     2,     3,     3,     1,     2,     1,     3,     2,
       2,     1,     1,     3,     1,     3,     1,     2,     1,     3,
       2,     2,     3,     3,     2,     2,     4,     3,     2,     1,
       3,     2,     2,     3,     2,     1,     2,     1,     3,     2,
       2,     3,     3,     2,     1
};


enum { YYENOMEM = -2 };

#define yyerrok         (yyerrstatus = 0)
#define yyclearin       (yychar = YYEMPTY)

#define YYACCEPT        goto yyacceptlab
#define YYABORT         goto yyabortlab
#define YYERROR         goto yyerrorlab
#define YYNOMEM         goto yyexhaustedlab


#define YYRECOVERING()  (!!yyerrstatus)

#define YYBACKUP(Token, Value)                                    \
  do                                                              \
    if (yychar == YYEMPTY)                                        \
      {                                                           \
        yychar = (Token);                                         \
        yylval = (Value);                                         \
        YYPOPSTACK (yylen);                                       \
        yystate = *yyssp;                                         \
        goto yybackup;                                            \
      }                                                           \
    else                                                          \
      {                                                           \
        yyerror (YY_("syntax error: cannot back up")); \
        YYERROR;                                                  \
      }                                                           \
  while (0)

/* Backward compatibility with an undocumented macro.
   Use YYerror or YYUNDEF. */
#define YYERRCODE YYUNDEF


/* Enable debugging if requested.  */
#if YYDEBUG

# ifndef YYFPRINTF
#  include <stdio.h> /* INFRINGES ON USER NAME SPACE */
#  define YYFPRINTF fprintf
# endif

# define YYDPRINTF(Args)                        \
do {                                            \
  if (yydebug)                                  \
    YYFPRINTF Args;                             \
} while (0)




# define YY_SYMBOL_PRINT(Title, Kind, Value, Location)                    \
do {                                                                      \
  if (yydebug)                                                            \
    {                                                                     \
      YYFPRINTF (stderr, "%s ", Title);                                   \
      yy_symbol_print (stderr,                                            \
                  Kind, Value); \
      YYFPRINTF (stderr, "\n");                                           \
    }                                                                     \
} while (0)


/*-----------------------------------.
| Print this symbol's value on YYO.  |
`-----------------------------------*/

static void
yy_symbol_value_print (FILE *yyo,
                       yysymbol_kind_t yykind, YYSTYPE const * const yyvaluep)
{
  FILE *yyoutput = yyo;
  YY_USE (yyoutput);
  if (!yyvaluep)
    return;
  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  YY_USE (yykind);
  YY_IGNORE_MAYBE_UNINITIALIZED_END
}


/*---------------------------.
| Print this symbol on YYO.  |
`---------------------------*/

static void
yy_symbol_print (FILE *yyo,
                 yysymbol_kind_t yykind, YYSTYPE const * const yyvaluep)
{
  YYFPRINTF (yyo, "%s %s (",
             yykind < YYNTOKENS ? "token" : "nterm", yysymbol_name (yykind));

  yy_symbol_value_print (yyo, yykind, yyvaluep);
  YYFPRINTF (yyo, ")");
}

/*------------------------------------------------------------------.
| yy_stack_print -- Print the state stack from its BOTTOM up to its |
| TOP (included).                                                   |
`------------------------------------------------------------------*/

static void
yy_stack_print (yy_state_t *yybottom, yy_state_t *yytop)
{
  YYFPRINTF (stderr, "Stack now");
  for (; yybottom <= yytop; yybottom++)
    {
      int yybot = *yybottom;
      YYFPRINTF (stderr, " %d", yybot);
    }
  YYFPRINTF (stderr, "\n");
}

# define YY_STACK_PRINT(Bottom, Top)                            \
do {                                                            \
  if (yydebug)                                                  \
    yy_stack_print ((Bottom), (Top));                           \
} while (0)


/*------------------------------------------------.
| Report that the YYRULE is going to be reduced.  |
`------------------------------------------------*/

static void
yy_reduce_print (yy_state_t *yyssp, YYSTYPE *yyvsp,
                 int yyrule)
{
  int yylno = yyrline[yyrule];
  int yynrhs = yyr2[yyrule];
  int yyi;
  YYFPRINTF (stderr, "Reducing stack by rule %d (line %d):\n",
             yyrule - 1, yylno);
  /* The symbols being reduced.  */
  for (yyi = 0; yyi < yynrhs; yyi++)
    {
      YYFPRINTF (stderr, "   $%d = ", yyi + 1);
      yy_symbol_print (stderr,
                       YY_ACCESSING_SYMBOL (+yyssp[yyi + 1 - yynrhs]),
                       &yyvsp[(yyi + 1) - (yynrhs)]);
      YYFPRINTF (stderr, "\n");
    }
}

# define YY_REDUCE_PRINT(Rule)          \
do {                                    \
  if (yydebug)                          \
    yy_reduce_print (yyssp, yyvsp, Rule); \
} while (0)

/* Nonzero means print parse trace.  It is left uninitialized so that
   multiple parsers can coexist.  */
int yydebug;
#else /* !YYDEBUG */
# define YYDPRINTF(Args) ((void) 0)
# define YY_SYMBOL_PRINT(Title, Kind, Value, Location)
# define YY_STACK_PRINT(Bottom, Top)
# define YY_REDUCE_PRINT(Rule)
#endif /* !YYDEBUG */


/* YYINITDEPTH -- initial size of the parser's stacks.  */
#ifndef YYINITDEPTH
# define YYINITDEPTH 200
#endif

/* YYMAXDEPTH -- maximum size the stacks can grow to (effective only
   if the built-in stack extension method is used).

   Do not make this value too large; the results are undefined if
   YYSTACK_ALLOC_MAXIMUM < YYSTACK_BYTES (YYMAXDEPTH)
   evaluated with infinite-precision integer arithmetic.  */

#ifndef YYMAXDEPTH
# define YYMAXDEPTH 10000
#endif






/*-----------------------------------------------.
| Release the memory associated to this symbol.  |
`-----------------------------------------------*/

static void
yydestruct (const char *yymsg,
            yysymbol_kind_t yykind, YYSTYPE *yyvaluep)
{
  YY_USE (yyvaluep);
  if (!yymsg)
    yymsg = "Deleting";
  YY_SYMBOL_PRINT (yymsg, yykind, yyvaluep, yylocationp);

  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  YY_USE (yykind);
  YY_IGNORE_MAYBE_UNINITIALIZED_END
}


/* Lookahead token kind.  */
int yychar;

/* The semantic value of the lookahead symbol.  */
YYSTYPE yylval;
/* Number of syntax errors so far.  */
int yynerrs;




/*----------.
| yyparse.  |
`----------*/

int
yyparse (void)
{
    yy_state_fast_t yystate = 0;
    /* Number of tokens to shift before error messages enabled.  */
    int yyerrstatus = 0;

    /* Refer to the stacks through separate pointers, to allow yyoverflow
       to reallocate them elsewhere.  */

    /* Their size.  */
    YYPTRDIFF_T yystacksize = YYINITDEPTH;

    /* The state stack: array, bottom, top.  */
    yy_state_t yyssa[YYINITDEPTH];
    yy_state_t *yyss = yyssa;
    yy_state_t *yyssp = yyss;

    /* The semantic value stack: array, bottom, top.  */
    YYSTYPE yyvsa[YYINITDEPTH];
    YYSTYPE *yyvs = yyvsa;
    YYSTYPE *yyvsp = yyvs;

  int yyn;
  /* The return value of yyparse.  */
  int yyresult;
  /* Lookahead symbol kind.  */
  yysymbol_kind_t yytoken = YYSYMBOL_YYEMPTY;
  /* The variables used to return semantic value and location from the
     action routines.  */
  YYSTYPE yyval;



#define YYPOPSTACK(N)   (yyvsp -= (N), yyssp -= (N))

  /* The number of symbols on the RHS of the reduced rule.
     Keep to zero when no symbol should be popped.  */
  int yylen = 0;

  YYDPRINTF ((stderr, "Starting parse\n"));

  yychar = YYEMPTY; /* Cause a token to be read.  */

  goto yysetstate;


/*------------------------------------------------------------.
| yynewstate -- push a new state, which is found in yystate.  |
`------------------------------------------------------------*/
yynewstate:
  /* In all cases, when you get here, the value and location stacks
     have just been pushed.  So pushing a state here evens the stacks.  */
  yyssp++;


/*--------------------------------------------------------------------.
| yysetstate -- set current state (the top of the stack) to yystate.  |
`--------------------------------------------------------------------*/
yysetstate:
  YYDPRINTF ((stderr, "Entering state %d\n", yystate));
  YY_ASSERT (0 <= yystate && yystate < YYNSTATES);
  YY_IGNORE_USELESS_CAST_BEGIN
  *yyssp = YY_CAST (yy_state_t, yystate);
  YY_IGNORE_USELESS_CAST_END
  YY_STACK_PRINT (yyss, yyssp);

  if (yyss + yystacksize - 1 <= yyssp)
#if !defined yyoverflow && !defined YYSTACK_RELOCATE
    YYNOMEM;
#else
    {
      /* Get the current used size of the three stacks, in elements.  */
      YYPTRDIFF_T yysize = yyssp - yyss + 1;

# if defined yyoverflow
      {
        /* Give user a chance to reallocate the stack.  Use copies of
           these so that the &'s don't force the real ones into
           memory.  */
        yy_state_t *yyss1 = yyss;
        YYSTYPE *yyvs1 = yyvs;

        /* Each stack pointer address is followed by the size of the
           data in use in that stack, in bytes.  This used to be a
           conditional around just the two extra args, but that might
           be undefined if yyoverflow is a macro.  */
        yyoverflow (YY_("memory exhausted"),
                    &yyss1, yysize * YYSIZEOF (*yyssp),
                    &yyvs1, yysize * YYSIZEOF (*yyvsp),
                    &yystacksize);
        yyss = yyss1;
        yyvs = yyvs1;
      }
# else /* defined YYSTACK_RELOCATE */
      /* Extend the stack our own way.  */
      if (YYMAXDEPTH <= yystacksize)
        YYNOMEM;
      yystacksize *= 2;
      if (YYMAXDEPTH < yystacksize)
        yystacksize = YYMAXDEPTH;

      {
        yy_state_t *yyss1 = yyss;
        union yyalloc *yyptr =
          YY_CAST (union yyalloc *,
                   YYSTACK_ALLOC (YY_CAST (YYSIZE_T, YYSTACK_BYTES (yystacksize))));
        if (! yyptr)
          YYNOMEM;
        YYSTACK_RELOCATE (yyss_alloc, yyss);
        YYSTACK_RELOCATE (yyvs_alloc, yyvs);
#  undef YYSTACK_RELOCATE
        if (yyss1 != yyssa)
          YYSTACK_FREE (yyss1);
      }
# endif

      yyssp = yyss + yysize - 1;
      yyvsp = yyvs + yysize - 1;

      YY_IGNORE_USELESS_CAST_BEGIN
      YYDPRINTF ((stderr, "Stack size increased to %ld\n",
                  YY_CAST (long, yystacksize)));
      YY_IGNORE_USELESS_CAST_END

      if (yyss + yystacksize - 1 <= yyssp)
        YYABORT;
    }
#endif /* !defined yyoverflow && !defined YYSTACK_RELOCATE */


  if (yystate == YYFINAL)
    YYACCEPT;

  goto yybackup;


/*-----------.
| yybackup.  |
`-----------*/
yybackup:
  /* Do appropriate processing given the current state.  Read a
     lookahead token if we need one and don't already have one.  */

  /* First try to decide what to do without reference to lookahead token.  */
  yyn = yypact[yystate];
  if (yypact_value_is_default (yyn))
    goto yydefault;

  /* Not known => get a lookahead token if don't already have one.  */

  /* YYCHAR is either empty, or end-of-input, or a valid lookahead.  */
  if (yychar == YYEMPTY)
    {
      YYDPRINTF ((stderr, "Reading a token\n"));
      yychar = yylex ();
    }

  if (yychar <= YYEOF)
    {
      yychar = YYEOF;
      yytoken = YYSYMBOL_YYEOF;
      YYDPRINTF ((stderr, "Now at end of input.\n"));
    }
  else if (yychar == YYerror)
    {
      /* The scanner already issued an error message, process directly
         to error recovery.  But do not keep the error token as
         lookahead, it is too special and may lead us to an endless
         loop in error recovery. */
      yychar = YYUNDEF;
      yytoken = YYSYMBOL_YYerror;
      goto yyerrlab1;
    }
  else
    {
      yytoken = YYTRANSLATE (yychar);
      YY_SYMBOL_PRINT ("Next token is", yytoken, &yylval, &yylloc);
    }

  /* If the proper action on seeing token YYTOKEN is to reduce or to
     detect an error, take that action.  */
  yyn += yytoken;
  if (yyn < 0 || YYLAST < yyn || yycheck[yyn] != yytoken)
    goto yydefault;
  yyn = yytable[yyn];
  if (yyn <= 0)
    {
      if (yytable_value_is_error (yyn))
        goto yyerrlab;
      yyn = -yyn;
      goto yyreduce;
    }

  /* Count tokens shifted since error; after three, turn off error
     status.  */
  if (yyerrstatus)
    yyerrstatus--;

  /* Shift the lookahead token.  */
  YY_SYMBOL_PRINT ("Shifting", yytoken, &yylval, &yylloc);
  yystate = yyn;
  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  *++yyvsp = yylval;
  YY_IGNORE_MAYBE_UNINITIALIZED_END

  /* Discard the shifted token.  */
  yychar = YYEMPTY;
  goto yynewstate;


/*-----------------------------------------------------------.
| yydefault -- do the default action for the current state.  |
`-----------------------------------------------------------*/
yydefault:
  yyn = yydefact[yystate];
  if (yyn == 0)
    goto yyerrlab;
  goto yyreduce;


/*-----------------------------.
| yyreduce -- do a reduction.  |
`-----------------------------*/
yyreduce:
  /* yyn is the number of a rule to reduce with.  */
  yylen = yyr2[yyn];

  /* If YYLEN is nonzero, implement the default value of the action:
     '$$ = $1'.

     Otherwise, the following line sets YYVAL to garbage.
     This behavior is undocumented and Bison
     users should not rely upon it.  Assigning to YYVAL
     unconditionally makes the parser a bit smaller, and it avoids a
     GCC warning that YYVAL may be used uninitialized.  */
  yyval = yyvsp[1-yylen];


  YY_REDUCE_PRINT (yyn);
  switch (yyn)
    {
  case 4: /* section: JACOBIAN PARAMETER  */
#line 108 "scan.y"
                  { CmdJacobian( (yyvsp[0].str) );
                  }
#line 1509 "y.tab.c"
    break;

  case 5: /* section: HESSIAN PARAMETER  */
#line 111 "scan.y"
                  { CmdHessian( (yyvsp[0].str) );
                  }
#line 1516 "y.tab.c"
    break;

  case 6: /* section: DECLARE PARAMETER  */
#line 114 "scan.y"
                  { CmdDeclareValues( (yyvsp[0].str) );
                  }
#line 1523 "y.tab.c"
    break;

  case 7: /* section: STOICMAT PARAMETER  */
#line 117 "scan.y"
                  { CmdStoicmat( (yyvsp[0].str) );
                  }
#line 1530 "y.tab.c"
    break;

  case 8: /* section: DOUBLE PARAMETER  */
#line 120 "scan.y"
                  { CmdDouble( (yyvsp[0].str) );
                  }
#line 1537 "y.tab.c"
    break;

  case 9: /* section: REORDER PARAMETER  */
#line 123 "scan.y"
                  { CmdReorder( (yyvsp[0].str) );
                  }
#line 1544 "y.tab.c"
    break;

  case 10: /* section: MEX PARAMETER  */
#line 126 "scan.y"
                  { CmdMex( (yyvsp[0].str) );
                  }
#line 1551 "y.tab.c"
    break;

  case 11: /* section: DUMMYINDEX PARAMETER  */
#line 129 "scan.y"
                  { CmdDummyindex( (yyvsp[0].str) );
                  }
#line 1558 "y.tab.c"
    break;

  case 12: /* section: EQNTAGS PARAMETER  */
#line 132 "scan.y"
                  { CmdEqntags( (yyvsp[0].str) );
                  }
#line 1565 "y.tab.c"
    break;

  case 13: /* section: FUNCTION PARAMETER  */
#line 135 "scan.y"
                  { CmdFunction( (yyvsp[0].str) );
                  }
#line 1572 "y.tab.c"
    break;

  case 14: /* section: STOCHASTIC PARAMETER  */
#line 138 "scan.y"
                  { CmdStochastic( (yyvsp[0].str) );
                  }
#line 1579 "y.tab.c"
    break;

  case 15: /* section: ATOMDECL atomlist  */
#line 141 "scan.y"
                  {}
#line 1585 "y.tab.c"
    break;

  case 16: /* section: CHECK atomlist  */
#line 143 "scan.y"
                  {}
#line 1591 "y.tab.c"
    break;

  case 17: /* section: DEFVAR species  */
#line 145 "scan.y"
                  {}
#line 1597 "y.tab.c"
    break;

  case 18: /* section: DEFRAD species  */
#line 147 "scan.y"
                  {}
#line 1603 "y.tab.c"
    break;

  case 19: /* section: DEFFIX species  */
#line 149 "scan.y"
                  {}
#line 1609 "y.tab.c"
    break;

  case 20: /* section: SETVAR setspclist  */
#line 151 "scan.y"
                  {}
#line 1615 "y.tab.c"
    break;

  case 21: /* section: SETRAD setspclist  */
#line 153 "scan.y"
                  {}
#line 1621 "y.tab.c"
    break;

  case 22: /* section: SETFIX setspclist  */
#line 155 "scan.y"
                  {}
#line 1627 "y.tab.c"
    break;

  case 23: /* section: INITVALUES initvalues  */
#line 157 "scan.y"
                  {}
#line 1633 "y.tab.c"
    break;

  case 24: /* section: EQUATIONS equations  */
#line 159 "scan.y"
                  {}
#line 1639 "y.tab.c"
    break;

  case 25: /* section: FAMILIES families  */
#line 161 "scan.y"
                  {}
#line 1645 "y.tab.c"
    break;

  case 26: /* section: LUMP lumps  */
#line 163 "scan.y"
                  {}
#line 1651 "y.tab.c"
    break;

  case 27: /* section: LOOKAT lookatlist  */
#line 165 "scan.y"
                  {}
#line 1657 "y.tab.c"
    break;

  case 28: /* section: MONITOR monitorlist  */
#line 167 "scan.y"
                  {}
#line 1663 "y.tab.c"
    break;

  case 29: /* section: TRANSPORT translist  */
#line 169 "scan.y"
                  {}
#line 1669 "y.tab.c"
    break;

  case 30: /* section: CHECKALL  */
#line 171 "scan.y"
                  { CheckAll(); }
#line 1675 "y.tab.c"
    break;

  case 31: /* section: LOOKATALL  */
#line 173 "scan.y"
                  { LookAtAll(); }
#line 1681 "y.tab.c"
    break;

  case 32: /* section: TRANSPORTALL  */
#line 175 "scan.y"
                  { TransportAll(); }
#line 1687 "y.tab.c"
    break;

  case 33: /* section: WRITE_ATM  */
#line 177 "scan.y"
                  { WriteAtoms(); }
#line 1693 "y.tab.c"
    break;

  case 34: /* section: WRITE_SPC  */
#line 179 "scan.y"
                  { WriteSpecies(); }
#line 1699 "y.tab.c"
    break;

  case 35: /* section: WRITE_MAT  */
#line 181 "scan.y"
                  { WriteMatrices(); }
#line 1705 "y.tab.c"
    break;

  case 36: /* section: WRITE_OPT  */
#line 183 "scan.y"
                  { WriteOptions(); }
#line 1711 "y.tab.c"
    break;

  case 37: /* section: USE PARAMETER  */
#line 185 "scan.y"
                  { CmdUse( (yyvsp[0].str) ); }
#line 1717 "y.tab.c"
    break;

  case 38: /* section: LANGUAGE PARAMETER  */
#line 187 "scan.y"
                  { CmdLanguage( (yyvsp[0].str) ); }
#line 1723 "y.tab.c"
    break;

  case 39: /* section: INITIALIZE PARAMETER  */
#line 189 "scan.y"
                  { DefineInitializeNbr( (yyvsp[0].str) ); }
#line 1729 "y.tab.c"
    break;

  case 40: /* section: XGRID PARAMETER  */
#line 191 "scan.y"
                  { DefineXGrid( (yyvsp[0].str) ); }
#line 1735 "y.tab.c"
    break;

  case 41: /* section: YGRID PARAMETER  */
#line 193 "scan.y"
                  { DefineYGrid( (yyvsp[0].str) ); }
#line 1741 "y.tab.c"
    break;

  case 42: /* section: ZGRID PARAMETER  */
#line 195 "scan.y"
                  { DefineZGrid( (yyvsp[0].str) ); }
#line 1747 "y.tab.c"
    break;

  case 43: /* section: INLINE INLCTX inlinecode ENDINLINE  */
#line 197 "scan.y"
                  { 
		    AddInlineCode( (yyvsp[-2].str), InlineBuf );
                    free( InlineBuf );
		  }
#line 1756 "y.tab.c"
    break;

  case 44: /* section: INLINE error  */
#line 202 "scan.y"
                  { ParserErrorMessage(); }
#line 1762 "y.tab.c"
    break;

  case 45: /* section: INTFILE PARAMETER  */
#line 204 "scan.y"
                  { CmdIntegrator( (yyvsp[0].str) ); }
#line 1768 "y.tab.c"
    break;

  case 46: /* section: DRIVER PARAMETER  */
#line 206 "scan.y"
                  { CmdDriver( (yyvsp[0].str) ); }
#line 1774 "y.tab.c"
    break;

  case 47: /* section: RUN PARAMETER  */
#line 208 "scan.y"
                  { CmdRun( (yyvsp[0].str) ); }
#line 1780 "y.tab.c"
    break;

  case 48: /* section: USES uselist  */
#line 210 "scan.y"
                  {}
#line 1786 "y.tab.c"
    break;

  case 49: /* section: SPARSEDATA PARAMETER  */
#line 212 "scan.y"
                  { SparseData( (yyvsp[0].str) ); }
#line 1792 "y.tab.c"
    break;

  case 50: /* section: FLUX PARAMETER  */
#line 214 "scan.y"
                  { CmdFlux( (yyvsp[0].str) ); }
#line 1798 "y.tab.c"
    break;

  case 51: /* section: AUTOREDUCE PARAMETER  */
#line 216 "scan.y"
                  { CmdAutoReduce( (yyvsp[0].str) ); }
#line 1804 "y.tab.c"
    break;

  case 52: /* section: UPPERCASEF90 PARAMETER  */
#line 218 "scan.y"
                  { CmdUpperCaseF90( (yyvsp[0].str) ); }
#line 1810 "y.tab.c"
    break;

  case 53: /* section: MINVERSION PARAMETER  */
#line 220 "scan.y"
                  { CmdMinVersion( (yyvsp[0].str) ); }
#line 1816 "y.tab.c"
    break;

  case 54: /* semicolon: semicolon ';'  */
#line 223 "scan.y"
                  { ScanWarning("Unnecessary ';'");
                  }
#line 1823 "y.tab.c"
    break;

  case 58: /* atomlist: error semicolon  */
#line 230 "scan.y"
                  { ParserErrorMessage(); }
#line 1829 "y.tab.c"
    break;

  case 59: /* atomdef: ATOMID  */
#line 233 "scan.y"
                  { switch( crt_section ) {
                      case ATOMDECL: DeclareAtom( (yyvsp[0].str) ); break;
                      case CHECK:    SetAtomType( (yyvsp[0].str), DO_CHECK ); break;
                    }
                  }
#line 1839 "y.tab.c"
    break;

  case 62: /* lookatlist: error semicolon  */
#line 242 "scan.y"
                  { ParserErrorMessage(); }
#line 1845 "y.tab.c"
    break;

  case 63: /* lookatspc: LKTID  */
#line 245 "scan.y"
                  { AddLookAt( (yyvsp[0].str) );
                  }
#line 1852 "y.tab.c"
    break;

  case 66: /* monitorlist: error semicolon  */
#line 251 "scan.y"
                  { ParserErrorMessage(); }
#line 1858 "y.tab.c"
    break;

  case 67: /* monitorspc: MNIID  */
#line 254 "scan.y"
                  { AddMonitor( (yyvsp[0].str) );
                  }
#line 1865 "y.tab.c"
    break;

  case 70: /* translist: error semicolon  */
#line 260 "scan.y"
                  { ParserErrorMessage(); }
#line 1871 "y.tab.c"
    break;

  case 71: /* transspc: TPTID  */
#line 263 "scan.y"
                  { AddTransport( (yyvsp[0].str) );
                  }
#line 1878 "y.tab.c"
    break;

  case 74: /* uselist: error semicolon  */
#line 269 "scan.y"
                  { ParserErrorMessage(); }
#line 1884 "y.tab.c"
    break;

  case 75: /* usefile: USEID  */
#line 272 "scan.y"
                  { AddUseFile( (yyvsp[0].str) );
                  }
#line 1891 "y.tab.c"
    break;

  case 78: /* setspclist: error semicolon  */
#line 278 "scan.y"
                  { ParserErrorMessage(); }
#line 1897 "y.tab.c"
    break;

  case 79: /* setspcspc: SSPID  */
#line 281 "scan.y"
                  { switch( crt_section ) {
                      case SETVAR: SetSpcType( VAR_SPC, (yyvsp[0].str) ); break;
                      case SETRAD: SetSpcType( RAD_SPC, (yyvsp[0].str) ); break;
                      case SETFIX: SetSpcType( FIX_SPC, (yyvsp[0].str) ); break;
                    }
                  }
#line 1908 "y.tab.c"
    break;

  case 82: /* families: error semicolon  */
#line 291 "scan.y"
                  { ParserErrorMessage(); }
#line 1914 "y.tab.c"
    break;

  case 83: /* family: SPCSPC FAMCOLON members  */
#line 294 "scan.y"
                  { DeclareFamily( (yyvsp[-2].str) );
                  }
#line 1921 "y.tab.c"
    break;

  case 86: /* member: SPCNR SPCSPC  */
#line 304 "scan.y"
                  { AddMember( (yyvsp[0].str), (yyvsp[-1].str) );
                  }
#line 1928 "y.tab.c"
    break;

  case 87: /* member: SPCSPC  */
#line 307 "scan.y"
                  { AddMember( (yyvsp[0].str), "1" );
		    /*		    FinalizeFamily();*/
                  }
#line 1936 "y.tab.c"
    break;

  case 90: /* species: error semicolon  */
#line 314 "scan.y"
                  { ParserErrorMessage(); }
#line 1942 "y.tab.c"
    break;

  case 93: /* spcname: SPCSPC SPCEQUAL atoms  */
#line 320 "scan.y"
                  { switch( crt_section ) {
                      case DEFVAR: DeclareSpecies( VAR_SPC, (yyvsp[-2].str) ); break;
                      case DEFRAD: DeclareSpecies( RAD_SPC, (yyvsp[-2].str) ); break;
                      case DEFFIX: DeclareSpecies( FIX_SPC, (yyvsp[-2].str) ); break;
                    } 
                  }
#line 1953 "y.tab.c"
    break;

  case 94: /* spcdef: SPCSPC  */
#line 328 "scan.y"
                  { switch( crt_section ) {
                      case DEFVAR: DeclareSpecies( VAR_SPC, (yyvsp[0].str) ); break;
                      case DEFRAD: DeclareSpecies( RAD_SPC, (yyvsp[0].str) ); break;
                      case DEFFIX: DeclareSpecies( FIX_SPC, (yyvsp[0].str) ); break;
                    } 
                  }
#line 1964 "y.tab.c"
    break;

  case 97: /* atom: SPCNR SPCSPC  */
#line 339 "scan.y"
                  { AddAtom( (yyvsp[0].str), (yyvsp[-1].str) );
                  }
#line 1971 "y.tab.c"
    break;

  case 98: /* atom: SPCSPC  */
#line 342 "scan.y"
                  { AddAtom( (yyvsp[0].str), "1" );
                  }
#line 1978 "y.tab.c"
    break;

  case 101: /* initvalues: error semicolon  */
#line 348 "scan.y"
                  { ParserErrorMessage(); }
#line 1984 "y.tab.c"
    break;

  case 102: /* assignment: INISPC INIEQUAL INIVALUE  */
#line 351 "scan.y"
                  { AssignInitialValue( (yyvsp[-2].str), (yyvsp[0].str) ); }
#line 1990 "y.tab.c"
    break;

  case 105: /* equations: error semicolon  */
#line 356 "scan.y"
                  { ParserErrorMessage();
                    eqState = LHS; 
                  }
#line 1998 "y.tab.c"
    break;

  case 106: /* equation: eqntag lefths righths rate  */
#line 361 "scan.y"
                  { eqState = LHS;
                    StoreEquationRate( (yyvsp[0].str), (yyvsp[-3].str) ); 
                    CheckEquation();
                  }
#line 2007 "y.tab.c"
    break;

  case 107: /* equation: lefths righths rate  */
#line 366 "scan.y"
                  { eqState = LHS;
                    StoreEquationRate( (yyvsp[0].str), "          " ); 
                    CheckEquation();
                  }
#line 2016 "y.tab.c"
    break;

  case 108: /* rate: RATE rate  */
#line 371 "scan.y"
                  { strcpy( (yyval.str), (yyvsp[-1].str) );
                    strcat( (yyval.str), (yyvsp[0].str) ); 
                  }
#line 2024 "y.tab.c"
    break;

  case 109: /* rate: RATE  */
#line 375 "scan.y"
                  { strcpy( (yyval.str), (yyvsp[0].str) );
                  }
#line 2031 "y.tab.c"
    break;

  case 110: /* eqntag: EQNLESS EQNTAG EQNGREATER  */
#line 379 "scan.y"
                  { strcpy( (yyval.str), (yyvsp[-1].str) );
                  }
#line 2038 "y.tab.c"
    break;

  case 111: /* lefths: expresion EQNEQUAL  */
#line 383 "scan.y"
                  { eqState = RHS; }
#line 2044 "y.tab.c"
    break;

  case 112: /* righths: expresion EQNCOLON  */
#line 386 "scan.y"
                  { ProcessTerm( eqState, "+", "1", "RR" ); /*Add a prod/loss species as last prod.*/ 
		                eqState = RAT; }
#line 2051 "y.tab.c"
    break;

  case 113: /* expresion: expresion EQNSIGN term  */
#line 390 "scan.y"
                  { ProcessTerm( eqState, (yyvsp[-1].str), crt_coef, crt_term ); 
                  }
#line 2058 "y.tab.c"
    break;

  case 114: /* expresion: EQNSIGN term  */
#line 393 "scan.y"
                  { ProcessTerm( eqState, (yyvsp[-1].str), crt_coef, crt_term );
                  }
#line 2065 "y.tab.c"
    break;

  case 115: /* expresion: term  */
#line 396 "scan.y"
                  { ProcessTerm( eqState, "+", crt_coef, crt_term );
                  }
#line 2072 "y.tab.c"
    break;

  case 116: /* term: EQNCOEF EQNSPC  */
#line 400 "scan.y"
                  { strcpy( crt_term, (yyvsp[0].str) );
                    strcpy( crt_coef, (yyvsp[-1].str) );  
                  }
#line 2080 "y.tab.c"
    break;

  case 117: /* term: EQNSPC  */
#line 404 "scan.y"
                  { strcpy( crt_term, (yyvsp[0].str) );         
                    strcpy( crt_coef, "1" ); 
                  }
#line 2088 "y.tab.c"
    break;

  case 120: /* lumps: error semicolon  */
#line 411 "scan.y"
                  { ParserErrorMessage(); }
#line 2094 "y.tab.c"
    break;

  case 121: /* lump: LMPSPC LMPPLUS lump  */
#line 414 "scan.y"
                  { AddLumpSpecies( (yyvsp[-2].str) );
                  }
#line 2101 "y.tab.c"
    break;

  case 122: /* lump: LMPSPC LMPCOLON LMPSPC  */
#line 417 "scan.y"
                  {
                    AddLumpSpecies( (yyvsp[-2].str) );
                    CheckLump( (yyvsp[0].str) );  
                  }
#line 2110 "y.tab.c"
    break;

  case 123: /* inlinecode: inlinecode INCODE  */
#line 422 "scan.y"
                  {
		    InlineBuf = AppendString( InlineBuf, (yyvsp[0].str), &InlineLen, MAX_INLINE );
		  }
#line 2118 "y.tab.c"
    break;

  case 124: /* inlinecode: INCODE  */
#line 426 "scan.y"
                  {
		    InlineBuf = malloc( MAX_INLINE ); 
                    InlineLen = MAX_INLINE;
		    strcpy( InlineBuf, (yyvsp[0].str));
		  }
#line 2128 "y.tab.c"
    break;


#line 2132 "y.tab.c"

      default: break;
    }
  /* User semantic actions sometimes alter yychar, and that requires
     that yytoken be updated with the new translation.  We take the
     approach of translating immediately before every use of yytoken.
     One alternative is translating here after every semantic action,
     but that translation would be missed if the semantic action invokes
     YYABORT, YYACCEPT, or YYERROR immediately after altering yychar or
     if it invokes YYBACKUP.  In the case of YYABORT or YYACCEPT, an
     incorrect destructor might then be invoked immediately.  In the
     case of YYERROR or YYBACKUP, subsequent parser actions might lead
     to an incorrect destructor call or verbose syntax error message
     before the lookahead is translated.  */
  YY_SYMBOL_PRINT ("-> $$ =", YY_CAST (yysymbol_kind_t, yyr1[yyn]), &yyval, &yyloc);

  YYPOPSTACK (yylen);
  yylen = 0;

  *++yyvsp = yyval;

  /* Now 'shift' the result of the reduction.  Determine what state
     that goes to, based on the state we popped back to and the rule
     number reduced by.  */
  {
    const int yylhs = yyr1[yyn] - YYNTOKENS;
    const int yyi = yypgoto[yylhs] + *yyssp;
    yystate = (0 <= yyi && yyi <= YYLAST && yycheck[yyi] == *yyssp
               ? yytable[yyi]
               : yydefgoto[yylhs]);
  }

  goto yynewstate;


/*--------------------------------------.
| yyerrlab -- here on detecting error.  |
`--------------------------------------*/
yyerrlab:
  /* Make sure we have latest lookahead translation.  See comments at
     user semantic actions for why this is necessary.  */
  yytoken = yychar == YYEMPTY ? YYSYMBOL_YYEMPTY : YYTRANSLATE (yychar);
  /* If not already recovering from an error, report this error.  */
  if (!yyerrstatus)
    {
      ++yynerrs;
      yyerror (YY_("syntax error"));
    }

  if (yyerrstatus == 3)
    {
      /* If just tried and failed to reuse lookahead token after an
         error, discard it.  */

      if (yychar <= YYEOF)
        {
          /* Return failure if at end of input.  */
          if (yychar == YYEOF)
            YYABORT;
        }
      else
        {
          yydestruct ("Error: discarding",
                      yytoken, &yylval);
          yychar = YYEMPTY;
        }
    }

  /* Else will try to reuse lookahead token after shifting the error
     token.  */
  goto yyerrlab1;


/*---------------------------------------------------.
| yyerrorlab -- error raised explicitly by YYERROR.  |
`---------------------------------------------------*/
yyerrorlab:
  /* Pacify compilers when the user code never invokes YYERROR and the
     label yyerrorlab therefore never appears in user code.  */
  if (0)
    YYERROR;
  ++yynerrs;

  /* Do not reclaim the symbols of the rule whose action triggered
     this YYERROR.  */
  YYPOPSTACK (yylen);
  yylen = 0;
  YY_STACK_PRINT (yyss, yyssp);
  yystate = *yyssp;
  goto yyerrlab1;


/*-------------------------------------------------------------.
| yyerrlab1 -- common code for both syntax error and YYERROR.  |
`-------------------------------------------------------------*/
yyerrlab1:
  yyerrstatus = 3;      /* Each real token shifted decrements this.  */

  /* Pop stack until we find a state that shifts the error token.  */
  for (;;)
    {
      yyn = yypact[yystate];
      if (!yypact_value_is_default (yyn))
        {
          yyn += YYSYMBOL_YYerror;
          if (0 <= yyn && yyn <= YYLAST && yycheck[yyn] == YYSYMBOL_YYerror)
            {
              yyn = yytable[yyn];
              if (0 < yyn)
                break;
            }
        }

      /* Pop the current state because it cannot handle the error token.  */
      if (yyssp == yyss)
        YYABORT;


      yydestruct ("Error: popping",
                  YY_ACCESSING_SYMBOL (yystate), yyvsp);
      YYPOPSTACK (1);
      yystate = *yyssp;
      YY_STACK_PRINT (yyss, yyssp);
    }

  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  *++yyvsp = yylval;
  YY_IGNORE_MAYBE_UNINITIALIZED_END


  /* Shift the error token.  */
  YY_SYMBOL_PRINT ("Shifting", YY_ACCESSING_SYMBOL (yyn), yyvsp, yylsp);

  yystate = yyn;
  goto yynewstate;


/*-------------------------------------.
| yyacceptlab -- YYACCEPT comes here.  |
`-------------------------------------*/
yyacceptlab:
  yyresult = 0;
  goto yyreturnlab;


/*-----------------------------------.
| yyabortlab -- YYABORT comes here.  |
`-----------------------------------*/
yyabortlab:
  yyresult = 1;
  goto yyreturnlab;


/*-----------------------------------------------------------.
| yyexhaustedlab -- YYNOMEM (memory exhaustion) comes here.  |
`-----------------------------------------------------------*/
yyexhaustedlab:
  yyerror (YY_("memory exhausted"));
  yyresult = 2;
  goto yyreturnlab;


/*----------------------------------------------------------.
| yyreturnlab -- parsing is finished, clean up and return.  |
`----------------------------------------------------------*/
yyreturnlab:
  if (yychar != YYEMPTY)
    {
      /* Make sure we have latest lookahead translation.  See comments at
         user semantic actions for why this is necessary.  */
      yytoken = YYTRANSLATE (yychar);
      yydestruct ("Cleanup: discarding lookahead",
                  yytoken, &yylval);
    }
  /* Do not reclaim the symbols of the rule whose action triggered
     this YYABORT or YYACCEPT.  */
  YYPOPSTACK (yylen);
  YY_STACK_PRINT (yyss, yyssp);
  while (yyssp != yyss)
    {
      yydestruct ("Cleanup: popping",
                  YY_ACCESSING_SYMBOL (+*yyssp), yyvsp);
      YYPOPSTACK (1);
    }
#ifndef yyoverflow
  if (yyss != yyssa)
    YYSTACK_FREE (yyss);
#endif

  return yyresult;
}

#line 432 "scan.y"


void yyerror( char * str )
{
}

void ParserErrorMessage()
{
  /* yyerrok; */
/*
  Message("[%d,%s] -> [%d,%s]", crtTokType, crtToken, nextTokType, nextToken );  
*/
  if( crtToken[0] == ';' ) {
    ParserError("Misplaced ';'");
    return;
  }
  switch( crtTokType ) {
    case ATOMID:
      ParserError("Missing ';' after '%s'", crtToken );
      break; 

    case SPCSPC: 
      ParserError("Missing ';' or '+' after '%s'", crtToken );
      break; 
    case SPCNR:
      ParserError("Missing species after '%s'", crtToken );
      break; 
    case SPCPLUS:
      ParserError("Missing atom after '%s'", crtToken );
      break; 
    case SPCEQUAL:
      ParserError("Invalid '=' after '%s'", crtToken );
      break; 

    case INISPC: 
      ParserError("Missing '=' after '%s'", crtToken );
      break; 
    case INIEQUAL: 
      ParserError("Missing value after '%s'", crtToken );
      break; 
    case INIVALUE: 
      ParserError("Missing ';' after '%s'", crtToken );
      break; 

    case EQNSPC: 
      ParserError("Missing '+' or '=' after '%s'", crtToken );
      break; 
    case EQNEQUAL: 
      ParserError("Invalid right hand side of equation");
      break; 
    case EQNCOLON: 
      ParserError("Missing rate after '%s'", crtToken );
      break; 
    case EQNSIGN: 
      ParserError("Missing coeficient after '%s'", crtToken );
      break; 
    case EQNCOEF: 
      ParserError("Missing species after '%s'", crtToken );
      break; 
    case RATE: 
      ParserError("Missing ';' after '%s'", crtToken );
      break; 

    case LMPSPC: 
      ParserError("Missing '+' or ':' or ';' after '%s'", crtToken );
      break; 
    case LMPPLUS: 
      ParserError("Missing species after '%s'", crtToken );
      break; 
    case LMPCOLON: 
      ParserError("Missing species after '%s'", crtToken );
      break; 
    case INLINE:
      ParserError("Missing inline option after '%s'", crtToken );
      break;

    default:
      ParserError("Syntax error after '%s'", crtToken ); 
  }
}


int Parser( char * filename )
{
extern int yydebug;
FILE *f;

  crt_filename = filename;

  f = fopen( crt_filename, "r" );
  if( f == 0 ) {
    FatalError(7,"%s: File not found", crt_filename);
  } 
  
  yyin = f;
  nError   = 0;
  nWarning = 0;
  yydebug = 0;

  yyparse();

  fclose( f );

  return nError;
}          

